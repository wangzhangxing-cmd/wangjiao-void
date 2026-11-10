"""Compare elastoplastic-rock FE (Abaqus v5) with the coupled semi-analytical run.

    python compare_ep_fe_sa.py 04_results/fem_v5/V5_B10_EP_coarse [more FE dirs ...]
           [--sa 04_results/sa_ep/B10_P2048]

Each FE dir is an extractor output folder (summary.csv + P0x0_* files); its mesh
label is taken from the folder name (coarse / medium / fine).  The SA folder is
the output of run_ep_sa.py.  Outputs go to 04_results/fem_v5/comparison_B10_EP:
history_comparison.csv (FE vs SA per pressure and mesh), distribution metrics at
5 and 10 MPa, and a summary figure.  Angle convention: local angle 0 = crown.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
QUANTITIES = (
    ("seal_mises_max_mpa", "seal peak von Mises (MPa)"),
    ("void_ur_mm", "mean void u_r (mm)"),
    ("contact_ur_mm", "mean contact u_r (mm)"),
    ("q_mean_resultant_mpa", "mean contact pressure (MPa)"),
    ("concrete_maxp_max_mpa", "RC max principal (MPa)"),
    ("plastic_area_m2", "plastic area (m2)"),
    ("plastic_outer_radius_m", "plastic outer radius (m)"),
)


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


def mesh_label(path):
    for name in ("coarse", "medium", "fine"):
        if name in path.name:
            return name
    return path.name


def fe_seal(fe_dir, step):
    rows = [r for r in read(fe_dir / ("%s_seal_surface.csv" % step)) if r["surface"] == "seal_inner"]
    phi = arr(rows, "local_angle_deg")
    o = np.argsort(phi)
    return phi[o], arr(rows, "mises_mpa")[o]


def sa_seal(sa_dir, step):
    rows = read(sa_dir / ("%s_seal_surface.csv" % step))
    return arr(rows, "local_angle_deg"), arr(rows, "inner_mises_mpa")


def fe_ur(fe_dir, step):
    rows = read(fe_dir / ("%s_interface.csv" % step))
    phi = arr(rows, "local_angle_deg")
    o = np.argsort(phi)
    return phi[o], arr(rows, "lining_ur_mm")[o]


def sa_ur(sa_dir, step):
    rows = read(sa_dir / ("%s_displacement.csv" % step))
    return arr(rows, "local_angle_deg"), arr(rows, "lining_ur_mm")


def plastic_cells(path, key):
    rows = [r for r in read(path) if r["x_m"] not in ("", "nan")]
    if not rows:
        return np.zeros((0, 2)), np.zeros(0)
    return np.column_stack([arr(rows, "x_m"), arr(rows, "y_m")]), arr(rows, key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fe_dirs", nargs="+")
    ap.add_argument("--sa", default=str(ROOT / "04_results" / "sa_ep" / "B10_P2048"))
    ap.add_argument("--out", default=str(ROOT / "04_results" / "fem_v5" / "comparison_B10_EP"))
    a = ap.parse_args()
    sa_dir, out = Path(a.sa), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fe_dirs = [Path(d) for d in a.fe_dirs]

    sa = {r["step"]: r for r in read(sa_dir / "summary.csv")}
    history = []
    for d in fe_dirs:
        for r in read(d / "summary.csv"):
            if r["step"] not in sa:
                continue
            row = dict(mesh=mesh_label(d), pressure_mpa=float(r["pressure_mpa"]))
            for key, _ in QUANTITIES:
                fe_v, sa_v = float(r[key]), float(sa[r["step"]][key])
                row["fe_" + key], row["sa_" + key] = fe_v, sa_v
                row["diff_" + key] = (100.0 * (fe_v / sa_v - 1.0)) if abs(sa_v) > 1e-12 else float("nan")
            history.append(row)
    write_csv(out / "history_comparison.csv", history)

    fields = []
    steps = [s for s in ("P050", "P100") if (sa_dir / ("%s_seal_surface.csv" % s)).exists()]
    for d in fe_dirs:
        for s in steps:
            for name, fe_f, sa_f in (("seal_inner_mises_mpa", fe_seal, sa_seal), ("lining_ur_mm", fe_ur, sa_ur)):
                pf, vf = fe_f(d, s)
                ps, vs = sa_f(sa_dir, s)
                vs_at = np.interp(pf, ps, vs, period=360.0)
                dv = vf - vs_at
                fields.append(dict(mesh=mesh_label(d), step=s, quantity=name,
                                   fe_max=float(vf.max()), sa_max=float(vs.max()),
                                   max_diff_percent=100.0 * (vf.max() / vs.max() - 1.0),
                                   fe_max_angle_deg=float(pf[np.argmax(vf)]), sa_max_angle_deg=float(ps[np.argmax(vs)]),
                                   rel_l2_percent=100.0 * float(np.sqrt(np.mean(dv ** 2)) / np.sqrt(np.mean(vs_at ** 2)))))
    if fields:
        write_csv(out / "field_comparison.csv", fields)
    (out / "comparison.json").write_text(json.dumps(dict(sa=str(sa_dir.relative_to(ROOT)),
                                                         fe=[str(d) for d in fe_dirs],
                                                         history=history, fields=fields), indent=1),
                                         encoding="utf-8")
    for r in fields:
        print(json.dumps(r), flush=True)
    plot(out, fe_dirs, sa_dir, sa, history, steps)


def plot(out, fe_dirs, sa_dir, sa, history, steps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 9})
    marks = {"coarse": ("s", "C3"), "medium": ("o", "C2"), "fine": ("^", "C4")}
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    p_sa = np.array([float(r["pressure_mpa"]) for r in sa.values()])
    sa_style = dict(color="C0", lw=1.4)

    def history_panel(a, key, ylabel, title):
        a.plot(p_sa, [float(r[key]) for r in sa.values()], "-", label="半解析", **sa_style)
        for d in fe_dirs:
            m = mesh_label(d)
            rows = [h for h in history if h["mesh"] == m]
            mk, c = marks.get(m, ("d", "C5"))
            a.plot([h["pressure_mpa"] for h in rows], [h["fe_" + key] for h in rows], ls="none", marker=mk,
                   mfc="none", mec=c, ms=5, label="有限元 " + m)
        a.set(xlabel="内压 p (MPa)", ylabel=ylabel, title=title, xlim=(0, 10.5))
        a.legend(loc="upper left")

    history_panel(ax[0, 0], "seal_mises_max_mpa", "钢板 von Mises 峰值 (MPa)", "(a) 钢板峰值应力")
    ax[0, 0].axhline(420, color="k", ls="--", lw=0.8)
    ax[0, 0].text(6.6, 428, "$f_y$ = 420 MPa", fontsize=8)
    history_panel(ax[0, 1], "void_ur_mm", "脱空段平均径向位移 (mm)", "(b) 脱空段位移")
    history_panel(ax[0, 2], "plastic_area_m2", "塑性区面积 (m$^2$)", "(c) 围岩塑性区面积")

    shade = dict(color="0.9", zorder=0)
    a = ax[1, 0]
    a.axvspan(-5, 5, **shade)
    for s, ls in zip(steps, ("--", "-")):
        ps, vs = sa_seal(sa_dir, s)
        a.plot(ps, vs, ls, color="C0", lw=1.3, label="半解析 %d MPa" % (int(s[1:]) // 10))
        for d in fe_dirs:
            m = mesh_label(d)
            mk, c = marks.get(m, ("d", "C5"))
            pf, vf = fe_seal(d, s)
            a.plot(pf[::2], vf[::2], ls="none", marker=mk, mfc="none", mec=c, ms=3, mew=0.7,
                   label="有限元 %s %d MPa" % (m, int(s[1:]) // 10))
    a.axhline(420, color="k", ls="--", lw=0.8)
    a.set(xlim=(-30, 30), ylim=(0, 460), xlabel="局部角 (°)", ylabel="钢板内表面 von Mises (MPa)",
          title="(d) 钢板内表面应力分布")
    a.legend(loc="lower left", fontsize=7)

    a = ax[1, 1]
    a.axvspan(-5, 5, **shade)
    s = steps[-1]
    ps, vs = sa_ur(sa_dir, s)
    a.plot(ps, vs, "-", label="半解析", **sa_style)
    for d in fe_dirs:
        m = mesh_label(d)
        mk, c = marks.get(m, ("d", "C5"))
        pf, vf = fe_ur(d, s)
        a.plot(pf[::2], vf[::2], ls="none", marker=mk, mfc="none", mec=c, ms=3, mew=0.7, label="有限元 " + m)
    a.set(xlim=(-45, 45), xlabel="局部角 (°)", ylabel="衬砌外表面径向位移 (mm)",
          title="(e) 衬砌径向位移, p = %d MPa" % (int(s[1:]) // 10))
    a.legend(loc="upper right")

    a = ax[1, 2]
    xy_sa, _ = plastic_cells(sa_dir / ("%s_rock_plastic_cells.csv" % s), "plastic_increment_norm")
    a.add_patch(plt.Circle((0, 0), 6.5, fill=False, color="k", lw=0.8))
    for d in fe_dirs:
        m = mesh_label(d)
        mk, c = marks.get(m, ("d", "C5"))
        xy, _ = plastic_cells(d / ("%s_rock_plastic_cells.csv" % s), "dpe_norm_elem_mean")
        a.plot(xy[:, 0], xy[:, 1], ls="none", marker=".", ms=1.0, color=c, alpha=0.5, label="有限元 " + m)
    a.plot(xy_sa[:, 0], xy_sa[:, 1], ls="none", marker=".", ms=1.5, color="C0", label="半解析")
    t = np.deg2rad(np.linspace(85, 95, 50))
    a.plot(6.5 * np.cos(t), 6.5 * np.sin(t), color="C1", lw=3, label="脱空段")
    lim = 1.05 * max(np.abs(xy_sa).max() if len(xy_sa) else 7.0, 7.0)
    a.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel="x (m)", ylabel="y (m)",
          title="(f) 塑性区, p = %d MPa" % (int(s[1:]) // 10))
    a.legend(loc="lower right", markerscale=6, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "B10_EP_FE_vs_SA.png", dpi=300)
    print("figure:", out / "B10_EP_FE_vs_SA.png")


if __name__ == "__main__":
    main()
