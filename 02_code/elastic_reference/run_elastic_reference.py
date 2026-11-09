"""Elastic-rock semi-analytical reference results (chapter 3 / chapter 4 support).

Elastic rock and the pre-cracked elastic lining make the problem linear and
history independent, so these runs take seconds and need none of the
plastic-kernel inputs.  Three tasks::

    python run_elastic_reference.py fem      # counterparts of V5_B10_EL / V5_B00_EL
    python run_elastic_reference.py panels   # interface-panel convergence, beta=10
    python run_elastic_reference.py sweep    # beta sweep, cracked vs uncracked lining
    python run_elastic_reference.py all

Outputs go to ``04_results/elastic_sa``.  Column names and the angle
convention (local angle 0 = crown, i.e. global theta - 90 deg) follow
``03_fem/fem_v5/extract_fem_results.py`` so FE and SA tables line up.
Displacements are increments after lining installation; q > 0 compresses the
rock wall.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "02_code" / "core"))

from layered_ring_operator import LiningInput, SealLiningOperator  # noqa: E402
from local_void_interface import FreeSectorInterface, arc_mean  # noqa: E402
from seal_stress_recovery import (  # noqa: E402
    _surface_coefficients, recover_concrete_face_stress, reconstruct_real, von_mises,
)

OUT = ROOT / "04_results" / "elastic_sa"
FY_MPA = 420.0
ROCK_E, ROCK_NU, R = 12.0e9, 0.26, 6.5
CRACKED = LiningInput(residual_hoop_ratio=0.05, residual_shear_ratio=0.05)
# Uncracked concrete: the hoop-reinforcement band cannot be represented with a
# hoop ratio > 1 in the operator, so bars are omitted (they add about 2% to the
# intact hoop stiffness).  Stated in the README.
UNCRACKED = LiningInput(residual_hoop_ratio=1.0, residual_shear_ratio=1.0,
                        include_reinforcement=False)
SAMPLES = 36000                     # 0.01 deg resolution for stress envelopes


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("%.9g" % v if isinstance(v, float) else v) for k, v in row.items()})


def local_angle(theta_local_rad: np.ndarray) -> np.ndarray:
    return (np.rad2deg(theta_local_rad) + 180.0) % 360.0 - 180.0


def seal_fields(interface, lining_u, pressure_pa, count=SAMPLES):
    """Seal inner/outer surface stresses on a uniform grid via inverse FFT."""
    coefficients = _surface_coefficients(interface, lining_u, pressure_pa)
    theta = np.arange(count) * 2.0 * np.pi / count
    out = {}
    for name, c in coefficients.items():
        spectrum = np.zeros((count // 2 + 1, 4), complex)
        spectrum[:len(c)] = c * count / 2.0
        spectrum[0] = c[0] * count
        stress = np.fft.irfft(spectrum, n=count, axis=0)
        out[name] = (stress, von_mises(stress))
    return theta, out


def seal_metrics(theta, fields, beta_deg):
    env = np.maximum(fields["inner"][1], fields["outer"][1]) / 1e6
    phi = local_angle(theta)
    i = int(np.argmax(env))
    result = dict(seal_mises_max_mpa=float(env[i]), seal_peak_local_angle_deg=float(phi[i]),
                  seal_peak_surface="inner" if fields["inner"][1][i] >= fields["outer"][1][i] else "outer")
    if beta_deg > 0:
        distance = np.abs(np.abs(phi) - beta_deg / 2.0)
        for d in (1.0, 2.0, 5.0):
            result["seal_mises_max_beyond_%gdeg_mpa" % d] = float(np.max(env[distance > d]))
    # membrane / bending split at the peak angle (hoop stress)
    inner_tt, outer_tt = fields["inner"][0][i, 1], fields["outer"][0][i, 1]
    result["seal_hoop_membrane_at_peak_mpa"] = float(0.5 * (inner_tt + outer_tt) / 1e6)
    result["seal_hoop_bending_at_peak_mpa"] = float(0.5 * (inner_tt - outer_tt) / 1e6)
    return result


def void_solution(beta_deg, pressure_pa, panels, core=1024, maximum=None, lining=CRACKED,
                  rock_E=ROCK_E, interface=None):
    if interface is None:
        maximum = maximum or max(8192, 8 * panels)
        interface = FreeSectorInterface(beta_deg, panels, core, maximum, rock_E=rock_E,
                                        rock_nu=ROCK_NU, lining_input=lining)
    result = interface.solve(pressure_pa)
    m = interface.metrics(result)
    theta, fields = seal_fields(interface, result["lining_u"], pressure_pa)
    q = result["panel_traction"]
    endpoint_q = q[np.argmin(interface.mid), 0], q[np.argmax(interface.mid), 0]
    row = dict(
        pressure_mpa=pressure_pa / 1e6,
        q_mean_mpa=m["mean_contact_pressure_mpa"],
        q_max_mpa=m["maximum_contact_pressure_mpa"],
        q_min_mpa=m["minimum_contact_pressure_mpa"],
        q_endpoint_panel_mpa=float(max(endpoint_q) / 1e6),
        contact_ur_mm=m["mean_contact_radial_displacement_mm"],
        void_ur_mm=m["mean_void_radial_displacement_mm"],
        **seal_metrics(theta, fields, beta_deg),
        linear_relative_residual=m["linear_relative_residual"],
        force_torque_balance_gap_pa=m["force_torque_balance_gap_pa"],
    )
    return interface, result, row, theta, fields


def axisymmetric_solution(pressure_pa, lining=CRACKED, rock_E=ROCK_E, helper=None):
    """beta = 0: exact axisymmetric coupling of the layered ring and elastic rock."""
    op = SealLiningOperator(lining)
    mode = op.mode(0)
    K, L = float(mode["K"][0, 0].real), float(mode["L"][0, 0].real)
    alpha, beta = -L / K, 1.0 / K
    G = rock_E / (2.0 * (1.0 + ROCK_NU))
    q = alpha * pressure_pa / (beta + R / (2.0 * G))
    u = R * q / (2.0 * G)
    helper = helper or FreeSectorInterface(10.0, 64, 64, 128, rock_E=rock_E, lining_input=lining)
    lining_u = np.zeros((1, 2), complex)
    lining_u[0, 0] = u
    theta, fields = seal_fields(helper, lining_u, pressure_pa, count=360)
    row = dict(pressure_mpa=pressure_pa / 1e6, q_mean_mpa=q / 1e6, q_max_mpa=q / 1e6,
               q_min_mpa=q / 1e6, q_endpoint_panel_mpa=q / 1e6, contact_ur_mm=u * 1e3,
               void_ur_mm=float("nan"), **seal_metrics(theta, fields, 0.0),
               linear_relative_residual=0.0, force_torque_balance_gap_pa=0.0)
    return row, helper, lining_u


# --------------------------------------------------------------------------
def task_fem():
    """SA counterparts of the elastic FE models, p = 1..10 MPa."""
    out = OUT / "fem_counterparts"
    out.mkdir(parents=True, exist_ok=True)
    pressures = [float(p) for p in range(1, 11)]
    summary = {}

    rows, helper = [], None
    for p in pressures:
        row, helper, lining_u = axisymmetric_solution(p * 1e6, helper=helper)
        rows.append(dict(step="P%03d" % int(10 * p), **row))
    write_csv(out / "SA_B00_EL_summary.csv", rows)
    summary["B00_EL"] = rows[-1]

    panels = 2048
    interface = FreeSectorInterface(10.0, panels, 1024, 8 * panels, lining_input=CRACKED)
    rows = []
    for p in pressures:
        interface_, result, row, theta, fields = void_solution(10.0, p * 1e6, panels, interface=interface)
        rows.append(dict(step="P%03d" % int(10 * p), panels=panels, **row))
        if p in (5.0, 10.0):
            export_distributions(out, "SA_B10_EL_P%03d" % int(10 * p), interface, result,
                                 theta, fields, p * 1e6)
    write_csv(out / "SA_B10_EL_summary.csv", rows)
    summary["B10_EL"] = rows[-1]
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)


def export_distributions(out, stem, interface, result, theta, fields, pressure_pa):
    phi = local_angle(theta)
    order = np.argsort(phi)
    step = max(1, len(theta) // 36000)
    rows = []
    for i in order[::step]:
        si, so = fields["inner"][0][i], fields["outer"][0][i]
        rows.append(dict(local_angle_deg=float(phi[i]),
                         inner_srr_mpa=si[0] / 1e6, inner_stt_mpa=si[1] / 1e6,
                         inner_szz_mpa=si[2] / 1e6, inner_srt_mpa=si[3] / 1e6,
                         inner_mises_mpa=fields["inner"][1][i] / 1e6,
                         outer_srr_mpa=so[0] / 1e6, outer_stt_mpa=so[1] / 1e6,
                         outer_szz_mpa=so[2] / 1e6, outer_srt_mpa=so[3] / 1e6,
                         outer_mises_mpa=fields["outer"][1][i] / 1e6))
    write_csv(out / (stem + "_seal_surface.csv"), rows)

    # lining outer-face displacement and rock-wall displacement, 0.05 deg grid
    grid = np.deg2rad(np.arange(-180.0, 180.0, 0.05))
    lin = reconstruct_real(result["lining_u"], grid)
    rock = reconstruct_real(result["rock_u"], grid)
    rows = [dict(local_angle_deg=float(np.rad2deg(t)), lining_ur_mm=lin[k, 0] * 1e3,
                 lining_ut_mm=lin[k, 1] * 1e3, rock_ur_mm=rock[k, 0] * 1e3,
                 rock_ut_mm=rock[k, 1] * 1e3) for k, t in enumerate(grid)]
    write_csv(out / (stem + "_displacement.csv"), rows)

    q = result["panel_traction"]
    rows = []
    for j in np.argsort(local_angle(interface.mid)):
        rows.append(dict(local_mid_deg=float(local_angle(np.array([interface.mid[j]]))[0]),
                         local_start_deg=float(local_angle(np.array([interface.edges[j]]))[0]),
                         local_end_deg=float(local_angle(np.array([interface.edges[j + 1]]))[0]),
                         q_mpa=q[j, 0] / 1e6, tau_mpa=q[j, 1] / 1e6))
    write_csv(out / (stem + "_interface_traction.csv"), rows)

    grid = np.deg2rad(np.arange(-180.0, 180.0, 0.05))
    faces = recover_concrete_face_stress(interface, result["lining_u"], pressure_pa, grid)
    maxp = np.max(np.column_stack([f["maximum_principal"] for f in faces.values()]), axis=1)
    rows = [dict(local_angle_deg=float(np.rad2deg(t)), max_principal_mpa=maxp[k] / 1e6)
            for k, t in enumerate(grid)]
    write_csv(out / (stem + "_concrete.csv"), rows)


# --------------------------------------------------------------------------
def task_panels():
    out = OUT / "panel_convergence"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for panels in (128, 256, 512, 1024, 2048, 4096, 8192):
        began = time.perf_counter()
        _, _, row, _, _ = void_solution(10.0, 10.0e6, panels)
        row = dict(panels=panels, maximum_order=max(8192, 8 * panels),
                   panel_width_deg=(360.0 - 10.0) / panels, **row,
                   elapsed_s=time.perf_counter() - began)
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("panels", "q_mean_mpa", "void_ur_mm",
                                               "seal_mises_max_mpa", "elapsed_s")}), flush=True)
    # Truncation checks at 1024 panels: interface order and seal core order.
    for core, maximum in ((2048, 8192), (1024, 16384), (1024, 4096)):
        _, _, row, _, _ = void_solution(10.0, 10.0e6, 1024, core=core, maximum=maximum)
        rows.append(dict(panels=1024, maximum_order=maximum, panel_width_deg=350.0 / 1024,
                         **row, elapsed_s=float("nan"), core_order=core))
    for row in rows:
        row.setdefault("core_order", 1024)
    write_csv(out / "panel_convergence_10MPa.csv", rows)

    # Richardson extrapolation on the three finest panel counts (h ratio 2).
    main = [r for r in rows if r["core_order"] == 1024 and r["maximum_order"] == max(8192, 8 * r["panels"])]
    main.sort(key=lambda r: r["panels"])
    report = {}
    for key in ("seal_mises_max_mpa", "void_ur_mm", "q_mean_mpa", "q_endpoint_panel_mpa",
                "seal_mises_max_beyond_1deg_mpa", "seal_mises_max_beyond_2deg_mpa"):
        f1, f2, f3 = (main[-3][key], main[-2][key], main[-1][key])   # coarse -> fine
        d1, d2 = f2 - f1, f3 - f2
        entry = dict(values_last_three=[f1, f2, f3])
        if d1 != 0 and d2 / d1 > 0 and abs(d2) < abs(d1):
            order = math.log(d1 / d2) / math.log(2.0)
            extrapolated = f3 + d2 / (2.0 ** order - 1.0)
            entry.update(observed_order=order, extrapolated=extrapolated,
                         finest_relative_error_percent=100.0 * (f3 / extrapolated - 1.0),
                         gci_fine_percent=1.25 * abs(d2 / f3) / (2.0 ** order - 1.0) * 100.0)
        else:
            entry["note"] = "non-monotone or not reducing; no extrapolation"
        report[key] = entry
    (out / "richardson.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1), flush=True)


# --------------------------------------------------------------------------
def task_sweep():
    """Unit-pressure solutions; linear elasticity gives p_y = fy / sigma_peak(1 MPa)."""
    out = OUT / "beta_sweep"
    out.mkdir(parents=True, exist_ok=True)
    betas = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30)
    rows = []
    for label, lining in (("cracked", CRACKED), ("uncracked", UNCRACKED)):
        base, _, _ = axisymmetric_solution(1.0e6, lining=lining)
        rows.append(dict(lining=label, beta_deg=0.0, panels=0, **base,
                         first_yield_pressure_mpa=FY_MPA / base["seal_mises_max_mpa"],
                         seal_amplification=1.0, void_ur_amplification=float("nan")))
        for beta in betas:
            panels = 2048
            _, _, row, _, _ = void_solution(float(beta), 1.0e6, panels, lining=lining)
            rows.append(dict(lining=label, beta_deg=float(beta), panels=panels, **row,
                             first_yield_pressure_mpa=FY_MPA / row["seal_mises_max_mpa"],
                             seal_amplification=row["seal_mises_max_mpa"] / base["seal_mises_max_mpa"],
                             void_ur_amplification=row["void_ur_mm"] / base["contact_ur_mm"]))
            print(json.dumps({k: rows[-1][k] for k in ("lining", "beta_deg", "seal_mises_max_mpa",
                                                       "first_yield_pressure_mpa")}), flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
    rows = [{k: r.get(k, float("nan")) for k in keys} for r in rows]
    write_csv(out / "beta_sweep_unit_pressure.csv", rows)

    # critical void angle at given pressures (linear interpolation of p_y(beta))
    crit = []
    for label in ("cracked", "uncracked"):
        sub = [r for r in rows if r["lining"] == label]
        b = np.array([r["beta_deg"] for r in sub])
        py = np.array([r["first_yield_pressure_mpa"] for r in sub])
        monotone = bool(np.all(np.diff(py) < 0))
        for p in (4.0, 6.0, 8.0, 10.0, 12.0):
            if py[0] <= p:
                bc = 0.0
            elif py[-1] > p:
                bc = float("nan")
            else:
                k = int(np.flatnonzero(py <= p)[0])
                bc = float(b[k - 1] + (py[k - 1] - p) * (b[k] - b[k - 1]) / (py[k - 1] - py[k]))
            crit.append(dict(lining=label, pressure_mpa=p, critical_beta_deg=bc,
                             p_y_monotone_in_beta=monotone))
    write_csv(out / "critical_beta_elastic_rock.csv", crit)
    print(json.dumps(crit, indent=1), flush=True)


def main():
    tasks = sys.argv[1:] or ["all"]
    if "all" in tasks:
        tasks = ["fem", "panels", "sweep"]
    for task in tasks:
        began = time.perf_counter()
        {"fem": task_fem, "panels": task_panels, "sweep": task_sweep}[task]()
        print("TASK %s done in %.0f s" % (task, time.perf_counter() - began), flush=True)


if __name__ == "__main__":
    main()
