"""Coupled semi-analytical run with elastoplastic rock, for the FE comparison.

    python run_ep_sa.py --beta 10 --panels 2048

Needs (not in git, see README): results/elastic_rock_kernel_nr64_nt512/kernel.npz
and results/nonaxisymmetric_excavation_nr64_nt512/installation_state.npz.

History: matched 64x512 excavation/installation state, then internal pressure
in 0.1 MPa increments to 10 MPa (halved on non-convergence, as in
run_chapter3_convergence.py).  Restartable: the latest accepted state is
checkpointed after every increment.

Outputs (04_results/sa_ep/B<beta>_P<panels>/):
* summary.csv            one row per integer pressure, columns as FE summary.csv
* P050_* / P100_*        seal surface stress, lining/rock displacement,
                         panel tractions, lining stresses incl. rebar, plastic cells
Full states are written to results/sa_ep_B<beta>_P<panels>/ (large, not in git).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code" / "core"))
sys.path.insert(0, str(ROOT / "02_code" / "elastic_reference"))

from coupled_void_nonassociated import CoupledVoidRockDomain  # noqa: E402
from local_void_interface import FreeSectorInterface, arc_mean  # noqa: E402
from nonaxisymmetric_nonassociated_domain import FullCircleRockDomain  # noqa: E402
from seal_stress_recovery import (  # noqa: E402
    recover_concrete_face_stress, recover_lining_section_response, reconstruct_real,
)
from run_elastic_reference import export_distributions, seal_fields, write_csv  # noqa: E402

THRESHOLD = 1.0e-6


def advance(domain, old, target, trace, depth=0):
    try:
        state = domain.step(old, target, max_iterations=800, source_tolerance=1e-9,
                            stress_tolerance_pa=500.0)
        trace.append(dict(pressure_mpa=target / 1e6, iterations=int(state["iterations"])))
        return state
    except RuntimeError as exc:
        if "Coupled void iteration failed" not in str(exc) or depth >= 6:
            raise
        mid = 0.5 * (old["internal_pressure_pa"] + target)
        print(json.dumps(dict(status="cutback", to_mpa=target / 1e6, mid_mpa=mid / 1e6)), flush=True)
        middle = advance(domain, old, mid, trace, depth + 1)
        return advance(domain, middle, target, trace, depth + 1)


def tensor_norm(values):
    return np.sqrt(np.sum(values[:, :3] ** 2, axis=1) + 2.0 * values[:, 3] ** 2)


def metrics(domain, interface, state, installation, kernel, beta):
    p = state["internal_pressure_pa"]
    b = state["boundary"]
    theta, fields = seal_fields(interface, b["lining_u"], p)
    env = np.maximum(fields["inner"][1], fields["outer"][1]) / 1e6
    i = int(np.argmax(env))
    local = (np.rad2deg(theta) + 180.0) % 360.0 - 180.0
    grid = np.linspace(0.0, 2.0 * np.pi, 7200, endpoint=False)
    faces = recover_concrete_face_stress(interface, b["lining_u"], p, grid)
    names = list(faces)
    allp = np.max(np.column_stack([faces[n]["maximum_principal"] for n in names]), axis=1)
    bands = {2, 4}
    matrix = np.max(np.column_stack([faces[n]["maximum_principal"] for n in names
                                     if int(n[5:7]) not in bands]), axis=1)
    sec = recover_lining_section_response(interface, b["lining_u"], p, grid)
    delta = tensor_norm(state["plastic"] - installation["plastic"])
    radii = np.asarray(kernel["radii"])
    nr, nt = len(radii) - 1, int(kernel["nt"])
    area = np.repeat(np.pi * (radii[1:] ** 2 - radii[:-1] ** 2) / nt, nt)
    active = delta > THRESHOLD
    outer = np.repeat(radii[1:], nt)
    q = b["panel_traction"]
    return dict(
        pressure_mpa=p / 1e6,
        q_mean_resultant_mpa=float(np.mean(q[:, 0]) / 1e6),
        contact_ur_mm=float(arc_mean(b["lining_u"], interface.alpha, 2 * np.pi - interface.alpha)[0] * 1e3),
        void_ur_mm=float(arc_mean(b["lining_u"], -interface.alpha, interface.alpha)[0] * 1e3),
        seal_mises_max_mpa=float(env[i]), seal_peak_local_angle_deg=float(local[i]),
        concrete_maxp_max_mpa=float(np.max(allp) / 1e6),
        matrix_maxp_max_mpa=float(np.max(matrix) / 1e6),
        rebar_inner_max_mpa=float(np.nanmax(sec["rebar_in"]) / 1e6),
        rebar_outer_max_mpa=float(np.nanmax(sec["rebar_out"]) / 1e6),
        plastic_area_m2=float(np.sum(area[active])),
        plastic_outer_radius_m=float(np.max(outer[active])) if np.any(active) else 0.0,
        plastic_increment_max=float(np.max(delta)),
        minimum_contact_pressure_mpa=float(np.min(q[:, 0]) / 1e6),
        plasticity_in_outermost_ring=bool(np.any(active.reshape(nr, nt)[-1])),
    ), (theta, fields, grid, allp, matrix, sec, delta, active, area)


def export(out, stem, interface, state, kernel, extra):
    theta, fields, grid, allp, matrix, sec, delta, active, area = extra
    p = state["internal_pressure_pa"]
    export_distributions(out, stem, interface, state["boundary"], theta, fields, p)
    local = (np.rad2deg(grid) + 180.0) % 360.0 - 180.0
    o = np.argsort(local)
    write_csv(out / (stem + "_lining.csv"), [
        dict(local_angle_deg=float(local[k]), max_principal_mpa=float(allp[k] / 1e6),
             matrix_max_principal_mpa=float(matrix[k] / 1e6),
             rebar_inner_stress_mpa=float(sec["rebar_in"][k] / 1e6),
             rebar_outer_stress_mpa=float(sec["rebar_out"][k] / 1e6)) for k in o])
    pts = np.asarray(kernel["points"])
    write_csv(out / (stem + "_rock_plastic_cells.csv"), [
        dict(x_m=float(-pts[k, 1]), y_m=float(pts[k, 0]), area_m2=float(area[k]),
             plastic_increment_norm=float(delta[k])) for k in np.flatnonzero(active)] or
        [dict(x_m=float("nan"), y_m=float("nan"), area_m2=0.0, plastic_increment_norm=0.0)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--panels", type=int, default=2048)
    ap.add_argument("--target", type=float, default=10.0)
    a = ap.parse_args()
    tag = "B%02d_P%d" % (int(round(a.beta)), a.panels)
    out = ROOT / "04_results" / "sa_ep" / tag
    states = ROOT / "results" / ("sa_ep_" + tag)
    out.mkdir(parents=True, exist_ok=True)
    states.mkdir(parents=True, exist_ok=True)

    kernel = dict(np.load(ROOT / "results" / "elastic_rock_kernel_nr64_nt512" / "kernel.npz"))
    rock = FullCircleRockDomain(kernel)
    with np.load(ROOT / "results" / "nonaxisymmetric_excavation_nr64_nt512" / "installation_state.npz") as f:
        installation = {k: f[k].copy() for k in ("plastic", "strain", "stress", "wall_displacement_m")}
    interface = FreeSectorInterface(a.beta, a.panels, 1024, max(8192, 8 * a.panels))
    domain = CoupledVoidRockDomain(rock, installation, interface, 1024)

    checkpoint = states / "latest.npz"
    log = states / "progress.json"
    rows, steps = [], []
    if checkpoint.exists() and log.exists():
        saved = np.load(checkpoint)
        state = {k: saved[k] for k in ("plastic", "strain", "stress")}
        state["internal_pressure_pa"] = float(saved["internal_pressure_pa"])
        progress = json.loads(log.read_text(encoding="utf-8"))
        rows, steps = progress["rows"], progress["steps"]
        print(json.dumps(dict(status="resumed", pressure_mpa=state["internal_pressure_pa"] / 1e6)), flush=True)
    else:
        state = domain.initial_state()
    began = time.perf_counter()
    for p in np.round(np.arange(0.1, a.target + 1e-9, 0.1), 10):
        if p * 1e6 <= state["internal_pressure_pa"] + 0.01:
            continue
        t0 = time.perf_counter()
        trace = []
        state = advance(domain, state, float(p * 1e6), trace)
        steps.append(dict(pressure_mpa=float(p), iterations=[t["iterations"] for t in trace],
                          seconds=time.perf_counter() - t0))
        np.savez_compressed(states / "latest.tmp.npz", plastic=state["plastic"], strain=state["strain"],
                            stress=state["stress"], internal_pressure_pa=np.array(state["internal_pressure_pa"]))
        (states / "latest.tmp.npz").replace(checkpoint)
        if abs(p - round(p)) < 1e-9:
            state["boundary"] = domain.interface(state["internal_pressure_pa"], state["plastic"])
            row, extra = metrics(domain, interface, state, installation, kernel, a.beta)
            row["step"] = "P%03d" % int(round(10 * p))
            rows.append(row)
            b = state["boundary"]
            np.savez_compressed(states / ("state_%02dMPa.npz" % int(round(p))), plastic=state["plastic"],
                                strain=state["strain"], stress=state["stress"],
                                panel_traction=b["panel_traction"], lining_u=b["lining_u"],
                                rock_u=b["rock_u"], internal_pressure_pa=np.array(state["internal_pressure_pa"]))
            if int(round(p)) in (5, 10):
                export(out, row["step"], interface, state, kernel, extra)
            write_csv(out / "summary.csv", [dict(step=r["step"], **{k: v for k, v in r.items() if k != "step"})
                                             for r in rows])
            print(json.dumps(dict(status="integer_state", **row)), flush=True)
        log.write_text(json.dumps(dict(rows=rows, steps=steps), indent=1), encoding="utf-8")
        print(json.dumps(dict(status="accepted", pressure_mpa=float(p), iterations=steps[-1]["iterations"],
                              seconds=round(steps[-1]["seconds"], 1),
                              elapsed_min=round((time.perf_counter() - began) / 60, 1))), flush=True)
    (out / "run_info.json").write_text(json.dumps(dict(
        beta_deg=a.beta, panels=a.panels, core_order=1024, interface_order=max(8192, 8 * a.panels),
        wall_order=1024, rock_mesh="64x512", increment_mpa=0.1, source_tolerance=1e-9,
        stress_tolerance_pa=500.0, plastic_threshold=THRESHOLD, steps=steps), indent=1), encoding="utf-8")
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
