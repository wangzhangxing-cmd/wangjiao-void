"""Maximum plastic-zone depth: threshold check and FE / SA comparison.

    python plastic_depth.py

Plastic-zone depth = max over cells (r_centroid) - R, R = 6.5 m, for the
envelope of the excavation plastic zone and the operation plastic zone:
    depth(p) = max(depth_excavation, depth_operation(p)).
A cell is plastic when the norm of its plastic strain (excavation: total
plastic strain; operation: increment after lining installation) exceeds a
threshold.  The threshold is a numerical tolerance for "plastic strain > 0",
checked here against the closed-form yield-front radius of the no-void case.

Needs results/ (SA states, kernel) and 04_results/fem_v5 FE folders.
Outputs 04_results/fem_v5/comparison_B10_EP/plastic_depth*.csv and a figure.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code" / "core"))
from axisymmetric_closed_form_reference import constants  # noqa: E402

R = 6.5
THRESHOLDS = (1e-6, 1e-5, 3e-5, 1e-4, 3e-4)
ROCK = {"radius_m": R, "rock_modulus_mpa": 12000.0, "rock_poisson": 0.26, "cohesion_mpa": 0.9,
        "phi_deg": 40.0, "dilation_deg": 10.0, "p0_mpa": 3.75}
FE = ROOT / "04_results" / "fem_v5"
OUT = FE / "comparison_B10_EP"


def read(path):
    with open(path, encoding="utf-8-sig") as h:
        return list(csv.DictReader(h))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("%.6g" % v if isinstance(v, float) else v) for k, v in r.items()})


def norm(v):
    return np.sqrt(np.sum(v[:, :3] ** 2, axis=1) + 2.0 * v[:, 3] ** 2)


def depth(r, value, t):
    a = value > t
    return float(r[a].max() - R) if np.any(a) else 0.0


def fe_cells(folder, step):
    rows = [x for x in read(folder / ("%s_rock_plastic_cells.csv" % step)) if x["x_m"] not in ("", "nan")]
    r = np.hypot([float(x["x_m"]) for x in rows], [float(x["y_m"]) for x in rows])
    return np.asarray(r), np.array([float(x["dpe_norm_elem_mean"]) for x in rows])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- 1. threshold check: no-void FE vs closed-form yield front
    rp_exc = constants(ROCK)["excavation_plastic_radius_m"]
    cf = {int(round(float(r["pressure_mpa"]))): float(r["cf_yield_front_radius_m"])
          for r in read(FE / "comparison_B00_EP" / "B00_EP_vs_closed_form.csv")}
    kernel = np.load(ROOT / "results" / "elastic_rock_kernel_nr64_nt512" / "kernel.npz")
    r_sa = np.hypot(kernel["points"][:, 0], kernel["points"][:, 1])
    install = np.load(ROOT / "results" / "nonaxisymmetric_excavation_nr64_nt512" / "installation_state.npz")["plastic"]
    e_exc = norm(install)
    check = [dict(case="excavation (SA, 64x512)", pressure_mpa=0.0, closed_form_depth_m=rp_exc - R,
                  **{"depth_%g" % t: depth(r_sa, e_exc, t) for t in THRESHOLDS})]
    for p in range(5, 11):
        r, v = fe_cells(FE / "V5_B00_EP_medium", "P%03d" % (10 * p))
        check.append(dict(case="operation, no void (FE medium)", pressure_mpa=float(p),
                          closed_form_depth_m=cf[p] - R, **{"depth_%g" % t: depth(r, v, t) for t in THRESHOLDS}))
    write_csv(OUT / "plastic_depth_threshold_check.csv", check)

    # ---- 2. B10: envelope depth, SA and FE
    states = ROOT / "results" / "sa_ep_B10_P2048"
    exc = {"SA": depth(r_sa, e_exc, 1e-6)}
    for m in ("coarse", "medium"):
        js = json.loads((FE / ("V5_B10_EP_%s" % m) / "summary.json").read_text(encoding="utf-8"))
        exc[m] = js["excavation"]["plastic_outer_radius_m"] - R   # same threshold 1e-6
    rows = []
    for p in range(1, 11):
        row = dict(pressure_mpa=float(p))
        plastic = np.load(states / ("state_%02dMPa.npz" % p))["plastic"]
        ops = {"SA": (r_sa, norm(plastic - install))}
        for m in ("coarse", "medium"):
            ops[m] = fe_cells(FE / ("V5_B10_EP_%s" % m), "P%03d" % (10 * p))
        for name, (r, v) in ops.items():
            op = depth(r, v, 1e-6)
            row["%s_excavation_m" % name] = exc[name]
            row["%s_operation_m" % name] = op
            row["%s_max_depth_m" % name] = max(exc[name], op)
            row["%s_max_depth_1e-5_m" % name] = max(exc[name], depth(r, v, 1e-5))
        rows.append(row)
    write_csv(OUT / "plastic_depth_B10.csv", rows)
    for r in check + rows:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, ensure_ascii=False))
    plot(check, rows)


def plot(check, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["WenQuanYi Zen Hei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 9})
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    a = ax[0]
    op = [c for c in check if c["pressure_mpa"] > 0]
    p = [c["pressure_mpa"] for c in op]
    a.plot(p, [c["closed_form_depth_m"] for c in op], "k-", lw=1.4, label="闭合解屈服前沿")
    for t, mk in zip(THRESHOLDS, "os^dv"):
        a.plot(p, [c["depth_%g" % t] for c in op], ls="--", marker=mk, mfc="none", ms=5, lw=0.8,
               label="有限元, 阈值 %g" % t)
    a.set(xlabel="内压 p (MPa)", ylabel="运营塑性区深度 (m)", title="(a) 无脱空：阈值与闭合解")
    a.legend(fontsize=7)
    a = ax[1]
    p = [r["pressure_mpa"] for r in rows]
    a.plot(p, [r["SA_max_depth_m"] for r in rows], "-", color="C0", lw=1.4, label="半解析")
    for m, mk, c in (("coarse", "s", "C3"), ("medium", "o", "C2")):
        a.plot(p, [r["%s_max_depth_m" % m] for r in rows], ls="none", marker=mk, mfc="none", mec=c, ms=5,
               label="有限元 " + m)
    a.axhline(rows[0]["SA_excavation_m"], color="0.5", ls=":", lw=0.9)
    a.text(1.0, rows[0]["SA_excavation_m"] + 0.06, "开挖塑性区深度", fontsize=8, color="0.4")
    a.set(xlabel="内压 p (MPa)", ylabel="塑性区最大深度 (m)", title="(b) 10° 脱空：max(开挖, 运营)")
    a.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "plastic_depth.png", dpi=300)


if __name__ == "__main__":
    main()
