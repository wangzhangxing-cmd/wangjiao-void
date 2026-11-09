"""Compare elastic-rock FE (Abaqus v5) with the semi-analytical solution.

    python compare_elastic_fe_sa.py <results_V5_B10_EL_medium> [--panels 8192]

The SA solution is evaluated directly at the FE node angles (no interpolation).
Outputs go to ``04_results/fem_v5/comparison_B10_EL``: metric tables (CSV/JSON)
and figures (PNG, 300 dpi).  Angle convention: local angle 0 = crown.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code" / "core"))

from local_void_interface import FreeSectorInterface, arc_mean  # noqa: E402
from seal_stress_recovery import (  # noqa: E402
    _surface_coefficients, recover_concrete_face_stress, reconstruct_real, von_mises,
)

BETA = 10.0
P_MPA = 10.0


def read(path):
    with open(path, encoding="utf-8-sig") as h:
        return list(csv.DictReader(h))


def arr(rows, key):
    return np.array([float(r[key]) for r in rows])


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.6g" % v if isinstance(v, float) else v) for k, v in r.items()})


def field_metrics(name, phi, fe, sa, window=None):
    mask = np.ones(len(phi), bool) if window is None else (np.abs(phi) <= window)
    d = fe[mask] - sa[mask]
    i_fe, i_sa = int(np.argmax(fe[mask])), int(np.argmax(sa[mask]))
    return dict(
        quantity=name, range="all" if window is None else "|angle|<=%g" % window,
        fe_max=float(fe[mask][i_fe]), sa_max=float(sa[mask][i_sa]),
        max_diff_percent=100.0 * (fe[mask][i_fe] / sa[mask][i_sa] - 1.0),
        fe_max_angle_deg=float(phi[mask][i_fe]), sa_max_angle_deg=float(phi[mask][i_sa]),
        rel_l2_percent=100.0 * float(np.sqrt(np.mean(d ** 2)) / np.sqrt(np.mean(sa[mask] ** 2))),
        max_abs_diff=float(np.max(np.abs(d))),
    )


def window_force_fe(phi, q, lo, hi):
    """Trapezoid integral of nodal traction over |angle| in [lo, hi] (both sides), MN/m."""
    total = 0.0
    for sign in (-1.0, 1.0):
        a, b = sorted((sign * lo, sign * hi))
        m = (phi >= a - 1e-9) & (phi <= b + 1e-9)
        x, y = np.deg2rad(phi[m]), q[m]
        total += float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))
    return total * 6.5


def window_force_sa(interface, q, lo, hi):
    edges = np.rad2deg(interface.edges)
    edges = (edges + 180.0) % 360.0 - 180.0
    start, end = edges[:-1], edges[:-1] + np.rad2deg(interface.width)
    total = 0.0
    for sign in (-1.0, 1.0):
        a, b = sorted((sign * lo, sign * hi))
        overlap = np.clip(np.minimum(end, b) - np.maximum(start, a), 0.0, None)
        total += float(np.sum(overlap * q[:, 0]))
    return np.deg2rad(total) * 6.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fe_dir")
    ap.add_argument("--panels", type=int, default=8192)
    ap.add_argument("--out", default=str(ROOT / "04_results" / "fem_v5" / "comparison_B10_EL"))
    a = ap.parse_args()
    fe_dir, out = Path(a.fe_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print("building SA interface with %d panels ..." % a.panels, flush=True)
    it = FreeSectorInterface(BETA, a.panels, 1024, max(8192, 8 * a.panels))
    res = it.solve(P_MPA * 1e6)

    seal = read(fe_dir / "P100_seal_surface.csv")
    inter = read(fe_dir / "P100_interface.csv")
    conc = read(fe_dir / "P100_concrete.csv")
    summary_fe = read(fe_dir / "summary.csv")[-1]

    metrics = []
    curves = {}
    # ---- seal surfaces
    coeff = _surface_coefficients(it, res["lining_u"], P_MPA * 1e6)
    for face, key in (("seal_inner", "inner"), ("seal_outer", "outer")):
        rows = [r for r in seal if r["surface"] == face]
        phi = arr(rows, "local_angle_deg")
        sa_stress = reconstruct_real(coeff[key], np.deg2rad(phi))
        sa_vm = von_mises(sa_stress) / 1e6
        fe_vm = arr(rows, "mises_mpa")
        curves[face] = (phi, fe_vm, sa_vm, arr(rows, "stt_mpa"), sa_stress[:, 1] / 1e6)
        metrics.append(field_metrics(face + "_mises_mpa", phi, fe_vm, sa_vm))
        metrics.append(field_metrics(face + "_mises_mpa", phi, fe_vm, sa_vm, window=30.0))
        metrics.append(field_metrics(face + "_hoop_mpa", phi, arr(rows, "stt_mpa"), sa_stress[:, 1] / 1e6))

    # ---- lining displacement
    phi = arr(inter, "local_angle_deg")
    sa_u = reconstruct_real(res["lining_u"], np.deg2rad(phi)) * 1e3
    fe_ur = arr(inter, "lining_ur_mm")
    curves["ur"] = (phi, fe_ur, sa_u[:, 0])
    metrics.append(field_metrics("lining_ur_mm", phi, fe_ur, sa_u[:, 0]))
    metrics.append(field_metrics("lining_ut_mm_abs", phi, np.abs(arr(inter, "lining_ut_mm")), np.abs(sa_u[:, 1])))

    # ---- concrete (max principal over all RC layer faces / nodes)
    phic = arr(conc, "local_angle_deg")
    faces = recover_concrete_face_stress(it, res["lining_u"], P_MPA * 1e6, np.deg2rad(phic))
    sa_c = np.max(np.column_stack([f["maximum_principal"] for f in faces.values()]), axis=1) / 1e6
    fe_c = arr(conc, "max_principal_mpa")
    curves["concrete"] = (phic, fe_c, sa_c)
    metrics.append(field_metrics("rc_max_principal_mpa", phic, fe_c, sa_c))

    # ---- interface traction: loads transferred near the void edges
    q_fe = arr(inter, "q_rock_side_mpa")
    void = arr(inter, "void") > 0.5
    q_sa = res["panel_traction"] / 1e6
    curves["q"] = (phi, q_fe, it, q_sa)
    windows = []
    half = BETA / 2.0
    for lo, hi in ((0.0, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, 15.0), (15.0, 45.0)):
        f_fe = window_force_fe(phi, q_fe, half + lo, half + hi)
        f_sa = window_force_sa(it, q_sa, half + lo, half + hi)
        windows.append(dict(window_beyond_endpoint_deg="%g-%g" % (lo, hi), fe_force_mn_per_m=f_fe,
                            sa_force_mn_per_m=f_sa, diff_percent=100.0 * (f_fe / f_sa - 1.0)))
    write_csv(out / "edge_load_transfer.csv", windows)

    # ---- scalar summary
    m = it.metrics(res)
    scal = []
    for label, fe_v, sa_v in (
        ("mean contact pressure (resultant) MPa", float(summary_fe["q_mean_resultant_mpa"]), m["mean_contact_pressure_mpa"]),
        ("mean contact u_r mm", float(summary_fe["contact_ur_mm"]), m["mean_contact_radial_displacement_mm"]),
        ("mean void u_r mm", float(summary_fe["void_ur_mm"]), m["mean_void_radial_displacement_mm"]),
        ("seal max von Mises MPa (surface nodes)", float(summary_fe["seal_mises_max_mpa"]),
         float(max(curves["seal_inner"][2].max(), curves["seal_outer"][2].max()))),
    ):
        scal.append(dict(quantity=label, fe=fe_v, sa=sa_v, diff_percent=100.0 * (fe_v / sa_v - 1.0)))
    scal.append(dict(quantity="seal max von Mises MPa (FE integration points)",
                     fe=float(summary_fe["seal_mises_ip_max_mpa"]), sa=scal[-1]["sa"],
                     diff_percent=100.0 * (float(summary_fe["seal_mises_ip_max_mpa"]) / scal[-1]["sa"] - 1.0)))
    write_csv(out / "scalar_comparison_10MPa.csv", scal)
    write_csv(out / "field_comparison_10MPa.csv", metrics)
    (out / "comparison.json").write_text(json.dumps(dict(
        sa_panels=a.panels, scalars=scal, fields=metrics, edge_windows=windows), indent=1), encoding="utf-8")
    for r in scal + metrics + windows:
        print(json.dumps(r, ensure_ascii=False), flush=True)
    plot(out, curves, void)


def plot(out, c, void):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 9})
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))
    shade = dict(color="0.9", zorder=0)
    fe_style = dict(ls="none", marker="o", mfc="none", mec="C3", ms=3.2, mew=0.7, zorder=2)
    sa_style = dict(color="C0", lw=1.3, zorder=3)

    def thin(phi, values, step):
        o = np.argsort(phi)
        return phi[o][::step], values[o][::step]

    a = ax[0, 0]
    phi, fe, sa, _, _ = c["seal_inner"]
    o = np.argsort(phi)
    a.axvspan(-5, 5, **shade)
    a.plot(*thin(phi, fe, 2), label="有限元", **fe_style)
    a.plot(phi[o], sa[o], "-", label="半解析", **sa_style)
    a.axhline(420, color="k", ls="--", lw=0.8)
    a.text(12, 428, "$f_y$ = 420 MPa", fontsize=8)
    a.set(xlim=(-30, 30), ylim=(0, 460), xlabel="局部角 (°)",
          ylabel="钢板内表面 von Mises 应力 (MPa)", title="(a) 钢板内表面应力")
    a.legend(loc="lower left")

    a = ax[0, 1]
    phi, fe, sa = c["ur"]
    o = np.argsort(phi)
    a.axvspan(-5, 5, **shade)
    a.plot(*thin(phi, fe, 2), label="有限元", **fe_style)
    a.plot(phi[o], sa[o], "-", label="半解析", **sa_style)
    a.set(xlim=(-30, 30), xlabel="局部角 (°)", ylabel="衬砌外表面径向位移 (mm)",
          title="(b) 衬砌径向位移")
    a.legend()

    a = ax[1, 0]
    phi, fe, it, q = c["q"]
    o = np.argsort(phi)
    edges = (np.rad2deg(it.edges) + 180.0) % 360.0 - 180.0
    a.axvspan(-5, 5, **shade)
    edges = edges[:-1]
    keep = (edges > -20.5) & (edges < 20.5)
    xs = np.c_[edges[keep], edges[keep] + np.rad2deg(it.width), np.full(keep.sum(), np.nan)].ravel()
    ys = np.c_[q[keep, 0], q[keep, 0], np.full(keep.sum(), np.nan)].ravel()
    a.plot(*thin(phi, fe, 1), label="有限元（岩侧节点应力）", **fe_style)
    a.plot(xs, ys, "-", label="半解析（分段常值牵引）", **sa_style)
    a.set(xlim=(-20, 20), ylim=(0, 40), xlabel="局部角 (°)", ylabel="界面法向压力 q (MPa)",
          title="(c) 界面压力（端点为奇异点，纵轴截断）")
    a.legend(loc="upper right")

    a = ax[1, 1]
    phi, fe, sa = c["concrete"]
    o = np.argsort(phi)
    a.axvspan(-5, 5, **shade)
    a.plot(*thin(phi, fe, 2), label="有限元", **fe_style)
    a.plot(phi[o], sa[o], "-", label="半解析", **sa_style)
    a.annotate("端点奇异点\n（有限元节点外推值）", xy=(-4.96, 91.9), xytext=(-27, 70), fontsize=8,
               arrowprops=dict(arrowstyle="->", lw=0.6))
    a.set(xlim=(-30, 30), xlabel="局部角 (°)", ylabel="衬砌最大主应力 (MPa)",
          title="(d) 衬砌最大主应力（拱顶峰值在外层钢筋增强带）")
    a.legend()
    fig.suptitle("弹性围岩、10° 拱顶脱空、内压 10 MPa：有限元与半解析对比（灰色为脱空区）")
    fig.tight_layout()
    fig.savefig(out / "B10_EL_10MPa_distributions.png", dpi=300)
    print("figure written", out / "B10_EL_10MPa_distributions.png")


if __name__ == "__main__":
    main()
