# -*- coding: utf-8 -*-
"""Extract chapter-3 validation quantities from a v5 ODB.

Run inside Abaqus Python (2022: Python 2.7; 2024+: Python 3)::

    abaqus python extract_fem_results.py --odb V5_B10_EP_medium.odb ^
        --meta V5_B10_EP_medium_meta.json --out results_V5_B10_EP_medium

or with no arguments (also Abaqus/CAE File > Run Script): every ``<job>.odb``
in the current work directory that has a ``<job>_meta.json`` next to it (or in
``models/``) is extracted into ``results_<job>/``.

All operating quantities are increments relative to the end of ADD-LINING,
the same reference state as the semi-analytical model.  Signs: q > 0 is
compression of the rock wall by the lining; displacement u_r > 0 is outward.

Per pressure step the script writes (``<step>`` = P010 ... P100):

* ``<step>_seal_surface.csv``  seal inner/outer face, every node: stresses
  extrapolated to the node (ELEMENT_NODAL, averaged over the elements sharing
  the node) and the von Mises stress; this is the quantity comparable to the
  semi-analytical surface stress.
* ``<step>_interface.csv``     lining outer face, every node: u_r, u_t of the
  lining and of the rock wall, normal/shear traction from the rock-side and the
  lining-side nodal stresses, void flag.
* ``<step>_concrete.csv``      per angle: max principal stress of all RC layers,
  of the cracked matrix layers only, and the physical bar stresses (band hoop
  strain x 200 GPa) of the inner and outer reinforcement.
* ``<step>_interface_nodal_force.csv`` lining outer-face nodal forces (exact
  traction resultants per node; use these for load transfer near the void).
* ``<step>_rock_plastic_cells.csv`` (EP only) elements with operational
  plastic strain increment norm > threshold.
* ``summary.json`` / ``summary.csv`` one row per step with all scalar metrics.
"""
from __future__ import print_function

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
from odbAccess import openOdb
from abaqusConstants import ELEMENT_NODAL, INTEGRATION_POINT

PY2 = sys.version_info[0] == 2
PLASTIC_THRESHOLD = 1.0e-6
REBAR_E = 200.0e9   # reinforcing steel modulus; bar stress = band hoop strain * REBAR_E


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def write_csv(path, header, rows):
    if PY2:
        handle = open(path, "wb")
    else:
        handle = open(path, "w", newline="")
    with handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["%.9g" % v if isinstance(v, float) else v for v in row])


def blocks(field):
    """Concatenate bulkDataBlocks -> (data, nodeLabels, elementLabels)."""
    data, nodes, elems = [], [], []
    for block in field.bulkDataBlocks:
        data.append(np.array(block.data, dtype=float))
        nl = block.nodeLabels
        el = block.elementLabels
        n = len(block.data)
        nodes.append(np.array(nl, dtype=int) if nl is not None and len(nl) else np.zeros(n, int))
        elems.append(np.array(el, dtype=int) if el is not None and len(el) else np.zeros(n, int))
    if not data:
        return np.zeros((0, 0)), np.zeros(0, int), np.zeros(0, int)
    return np.vstack(data), np.concatenate(nodes), np.concatenate(elems)


def nodal_vector(frame, region, name="U"):
    data, nodes, _ = blocks(frame.fieldOutputs[name].getSubset(region=region))
    return dict((int(n), data[k]) for k, n in enumerate(nodes))


def nodal_average(frame, region, name, wanted):
    """ELEMENT_NODAL values averaged per node (only nodes in ``wanted``).

    Returns (mean, max_mises_over_contributing_elements) dictionaries."""
    data, nodes, _ = blocks(frame.fieldOutputs[name].getSubset(
        region=region, position=ELEMENT_NODAL))
    wanted = set(wanted)
    total, count, vm_max = {}, {}, {}
    for k, n in enumerate(nodes):
        n = int(n)
        if n not in wanted:
            continue
        row = data[k]
        total[n] = total.get(n, 0.0) + row
        count[n] = count.get(n, 0) + 1
        if name == "S":
            vm_max[n] = max(vm_max.get(n, 0.0), mises(row))
    return dict((n, total[n] / count[n]) for n in total), vm_max


def mises(s):
    s11, s22, s33, s12 = s[0], s[1], s[2], s[3]
    return math.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2)
                     + 3.0 * s12 ** 2)


def max_principal(s):
    c, r = 0.5 * (s[0] + s[1]), math.sqrt((0.5 * (s[0] - s[1])) ** 2 + s[3] ** 2)
    return max(c + r, s[2])


def to_polar(s, theta):
    """Global (11, 22, 33, 12) stress -> (rr, tt, zz, rt)."""
    c, n = math.cos(theta), math.sin(theta)
    rr = c * c * s[0] + n * n * s[1] + 2.0 * c * n * s[3]
    tt = n * n * s[0] + c * c * s[1] - 2.0 * c * n * s[3]
    rt = (s[1] - s[0]) * n * c + s[3] * (c * c - n * n)
    return rr, tt, s[2], rt


def polar_u(u, theta):
    c, n = math.cos(theta), math.sin(theta)
    return u[0] * c + u[1] * n, -u[0] * n + u[1] * c


def arc_mean(phi_deg, values, lo, hi):
    """Trapezoidal mean over lo <= phi <= hi (degrees, may wrap past 180)."""
    phi = np.asarray(phi_deg, float)
    vals = np.asarray(values, float)
    order = np.argsort(phi)
    phi, vals = phi[order], vals[order]
    if hi > 180.0:            # contact arc [beta/2, 360-beta/2] in local angle
        phi = np.where(phi < lo - 1e-9, phi + 360.0, phi)
        order = np.argsort(phi)
        phi, vals = phi[order], vals[order]
    mask = (phi >= lo - 1e-9) & (phi <= hi + 1e-9)
    x, y = phi[mask], vals[mask]
    if len(x) < 2:
        return float("nan")
    integral = float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))
    return integral / float(x[-1] - x[0])


def element_geometry(instance):
    xy = dict((n.label, n.coordinates[:2]) for n in instance.nodes)
    geo = {}
    for e in instance.elements:
        c = e.connectivity
        ring = [c[0], c[4], c[1], c[5], c[2], c[6], c[3], c[7]] if len(c) == 8 else list(c)
        pts = np.array([xy[k] for k in ring], float)
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        geo[e.label] = (cx, cy, area)
    return geo


def plastic_by_element(frame, instance):
    """Element-mean plastic strain (11, 22, 33, 12 engineering) and per-IP list."""
    if "PE" not in frame.fieldOutputs.keys():
        return {}
    data, _, elems = blocks(frame.fieldOutputs["PE"].getSubset(
        region=instance, position=INTEGRATION_POINT))
    out = {}
    for k, e in enumerate(elems):
        out.setdefault(int(e), []).append(data[k])
    return dict((e, np.array(v)) for e, v in out.items())


def tensor_norm(pe):
    """Frobenius norm with tensor shear (Abaqus PE12 is engineering gamma)."""
    pe = np.atleast_2d(pe)
    return np.sqrt(pe[:, 0] ** 2 + pe[:, 1] ** 2 + pe[:, 2] ** 2 + 0.5 * pe[:, 3] ** 2)


def plain_str(value):
    """Python 2: json gives ``unicode``; Abaqus repositories need ``str`` keys."""
    if PY2 and isinstance(value, unicode):  # noqa: F821  (Python 2 only)
        return value.encode("ascii")
    if isinstance(value, dict):
        return dict((plain_str(k), plain_str(v)) for k, v in value.items())
    if isinstance(value, list):
        return [plain_str(v) for v in value]
    return value


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Extract v5 validation results")
    ap.add_argument("--odb")
    ap.add_argument("--meta")
    ap.add_argument("--out")
    argv = [v for v in sys.argv[1:] if v.split("=")[0] in ("--odb", "--meta", "--out")
            or not v.startswith("-")]
    args, _ = ap.parse_known_args(argv)
    if args.odb and args.meta:
        jobs = [(args.odb, args.meta, args.out or "results_" + os.path.splitext(
            os.path.basename(args.odb))[0])]
    else:
        jobs = []
        for name in sorted(os.listdir(os.getcwd())):
            if not name.lower().endswith(".odb"):
                continue
            job = name[:-4]
            for folder in (os.getcwd(), os.path.join(os.getcwd(), "models")):
                meta = os.path.join(folder, job + "_meta.json")
                if os.path.isfile(meta):
                    jobs.append((name, meta, "results_" + job))
                    break
            else:
                print("skip %s: no %s_meta.json" % (name, job))
        if not jobs:
            print("No ODB with a matching _meta.json in", os.getcwd())
    for odb_path, meta_path, out in jobs:
        print("extracting", odb_path)
        extract(str(odb_path), str(meta_path), str(out))


class _Args(object):
    def __init__(self, odb, meta, out):
        self.odb, self.meta, self.out = odb, meta, out


def extract(odb_path, meta_path, out):
    args = _Args(odb_path, meta_path, out)
    meta = plain_str(json.load(open(args.meta)))
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    beta = float(meta["void_angle_deg"])
    half = beta / 2.0
    phi = np.array(meta["surface_nodes"]["phi_deg"], float)
    theta = np.radians(phi + 90.0)
    sn = meta["surface_nodes"]
    is_void = (np.abs(phi) < half - 1e-9) if beta > 0 else np.zeros(len(phi), bool)
    contact_lo, contact_hi = half, 360.0 - half
    phi_sorted = np.sort(phi)

    odb = openOdb(path=args.odb, readOnly=True)
    asm = odb.rootAssembly
    liner, rin, rout = (asm.instances["LINER-1"], asm.instances["ROCKIN-1"],
                        asm.instances["ROCKOUT-1"])
    seal_set, rc_set = liner.elementSets["SEAL"], liner.elementSets["RC"]
    out_row = liner.elementSets["OUT_ROW"]
    wall_row = rin.elementSets["WALL_ROW"]

    base = odb.steps["ADD-LINING"].frames[-1]
    excav = odb.steps["EXCAVATE"].frames[-1]
    xy = dict((n.label, n.coordinates[:2]) for n in liner.nodes)
    u0_liner = nodal_vector(base, liner)
    u0_rock = nodal_vector(base, rin)
    rock_geo = {"ROCKIN-1": element_geometry(rin), "ROCKOUT-1": element_geometry(rout)}
    pe_base = {"ROCKIN-1": plastic_by_element(base, rin), "ROCKOUT-1": plastic_by_element(base, rout)}
    pe_exc = plastic_by_element(excav, rin)

    # excavation-stage plasticity (compare with the semi-analytical installation state)
    exc_area, exc_rmax = 0.0, meta["radii"]["lining_outer"]
    for e, ips in pe_exc.items():
        if tensor_norm(ips.mean(axis=0))[0] > PLASTIC_THRESHOLD:
            cx, cy, a = rock_geo["ROCKIN-1"][e]
            exc_area += a
            exc_rmax = max(exc_rmax, math.hypot(cx, cy))

    summaries = []
    available = [str(k) for k in odb.steps.keys()]
    for step in meta["pressure_steps"]:
        step = str(step)
        if step not in available:
            print("missing step", step)
            continue
        frame = odb.steps[step].frames[-1]
        p_mpa = float(meta["pressures_mpa"][meta["pressure_steps"].index(step)])
        u_l = nodal_vector(frame, liner)
        u_r = nodal_vector(frame, rin)

        # ---- seal surfaces
        seal_rows, vm_env = [], {}
        surface_peak = {}
        for face in ("seal_inner", "seal_outer"):
            labels = sn[face]
            mean, vmax = nodal_average(frame, seal_set, "S", labels)
            vm = []
            for k, n in enumerate(labels):
                rr, tt, zz, rt = to_polar(mean[n], theta[k])
                v = mises(mean[n])
                vm.append(v)
                seal_rows.append((face, int(n), float(phi[k]), float(phi[k] + 90.0) % 360.0,
                                  rr / 1e6, tt / 1e6, zz / 1e6, rt / 1e6, v / 1e6,
                                  vmax.get(n, float("nan")) / 1e6))
            vm = np.array(vm)
            i = int(np.argmax(vm))
            surface_peak[face] = dict(mises_mpa=float(vm[i] / 1e6), local_angle_deg=float(phi[i]),
                                      mises_unaveraged_max_mpa=float(max(vmax.values()) / 1e6))
            vm_env[face] = vm
        envelope = np.maximum(vm_env["seal_inner"], vm_env["seal_outer"])
        ip_data, _, _ = blocks(frame.fieldOutputs["S"].getSubset(
            region=seal_set, position=INTEGRATION_POINT))
        seal_ip_max = max(mises(r) for r in ip_data) / 1e6
        write_csv(os.path.join(args.out, step + "_seal_surface.csv"),
                  ("surface", "node", "local_angle_deg", "theta_deg", "srr_mpa", "stt_mpa",
                   "szz_mpa", "srt_mpa", "mises_mpa", "mises_unaveraged_max_mpa"), seal_rows)

        # ---- interface: displacements and tractions
        rock_s, _ = nodal_average(frame, wall_row, "S", sn["rock_wall"])
        lin_s, _ = nodal_average(frame, out_row, "S", sn["liner_outer"])   # local r, theta
        rows, ur_l, q_rock, q_lin, gap = [], [], [], [], 0.0
        for k, (nl, nr) in enumerate(zip(sn["liner_outer"], sn["rock_wall"])):
            ul = u_l[nl] - u0_liner[nl]
            ur = u_r[nr] - u0_rock[nr]
            lr_, lt_ = polar_u(ul, theta[k])
            rr_, rt_ = polar_u(ur, theta[k])
            srr, _, _, srt = to_polar(rock_s[nr], theta[k])
            qr, tr = -srr, srt
            ql, tl = -lin_s[nl][0], lin_s[nl][3]
            if not is_void[k]:      # bonded nodes, void endpoints included
                gap = max(gap, math.hypot(lr_ - rr_, lt_ - rt_))
            rows.append((int(nl), int(nr), float(phi[k]), float(phi[k] + 90.0) % 360.0,
                         int(is_void[k]), lr_ * 1e3, lt_ * 1e3, rr_ * 1e3, rt_ * 1e3,
                         qr / 1e6, tr / 1e6, ql / 1e6, tl / 1e6))
            ur_l.append(lr_)
            q_rock.append(qr)
            q_lin.append(ql)
        write_csv(os.path.join(args.out, step + "_interface.csv"),
                  ("liner_node", "rock_node", "local_angle_deg", "theta_deg", "void",
                   "lining_ur_mm", "lining_ut_mm", "rock_ur_mm", "rock_ut_mm",
                   "q_rock_side_mpa", "tau_rock_side_mpa", "q_lining_side_mpa",
                   "tau_lining_side_mpa"), rows)

        # resultant-based mean contact pressure: sum of lining nodal forces
        nf1, nodes1, _ = blocks(frame.fieldOutputs["NFORC1"].getSubset(
            region=out_row, position=ELEMENT_NODAL))
        nf2, _, _ = blocks(frame.fieldOutputs["NFORC2"].getSubset(
            region=out_row, position=ELEMENT_NODAL))
        index = dict((int(n), k) for k, n in enumerate(sn["liner_outer"]))
        fr_sum = 0.0
        for k, n in enumerate(nodes1):
            j = index.get(int(n))
            if j is None or is_void[j]:
                continue
            fr_sum += nf1[k, 0] * math.cos(theta[j]) + nf2[k, 0] * math.sin(theta[j])
        contact_len = meta["radii"]["lining_outer"] * math.radians(360.0 - beta)

        # ---- interface nodal forces (exact load transfer; sum = traction resultant)
        nodal_fr, nodal_ft = {}, {}
        for k, n in enumerate(nodes1):
            j = index.get(int(n))
            if j is None:
                continue
            fr = nf1[k, 0] * math.cos(theta[j]) + nf2[k, 0] * math.sin(theta[j])
            ft = -nf1[k, 0] * math.sin(theta[j]) + nf2[k, 0] * math.cos(theta[j])
            nodal_fr[j] = nodal_fr.get(j, 0.0) + fr
            nodal_ft[j] = nodal_ft.get(j, 0.0) + ft
        write_csv(os.path.join(args.out, step + "_interface_nodal_force.csv"),
                  ("liner_node", "local_angle_deg", "void", "radial_force_kn_per_m",
                   "tangential_force_kn_per_m"),
                  [(int(sn["liner_outer"][j]), float(phi[j]), int(is_void[j]),
                    nodal_fr.get(j, 0.0) / 1e3, nodal_ft.get(j, 0.0) / 1e3)
                   for j in np.argsort(phi)])

        # ---- lining: matrix concrete, reinforcement and all-layer envelope
        def per_angle(elsets, field, reduce_value):
            """Group ELEMENT_NODAL values by exact mesh angle -> list of values."""
            groups = {}
            for name in elsets:
                if name not in liner.elementSets.keys():
                    continue
                data, nodes, _ = blocks(frame.fieldOutputs[field].getSubset(
                    region=liner.elementSets[name], position=ELEMENT_NODAL))
                for k, n in enumerate(nodes):
                    x, y = xy[int(n)]
                    a = (math.degrees(math.atan2(y, x)) - 90.0 + 180.0) % 360.0 - 180.0
                    # Snap to the exact mesh angle: ODB coordinates are single
                    # precision, so nodes on one radial line differ slightly.
                    j = int(np.argmin(np.abs((phi_sorted - a + 180.0) % 360.0 - 180.0)))
                    groups.setdefault(float(phi_sorted[j]), []).append(reduce_value(data[k]))
            return groups

        rc_names = sorted(str(k) for k in liner.elementSets.keys()
                          if str(k).startswith("RC") and str(k)[2:].isdigit())
        bands = [r["name"] for r in meta["rc_layers"] if "reinforcement" in r["label"]]
        band_inner = [r["name"] for r in meta["rc_layers"] if r["label"].startswith("inner")]
        band_outer = [r["name"] for r in meta["rc_layers"] if r["label"].startswith("outer")]
        matrix = [n for n in rc_names if n not in bands]
        envelope_all = per_angle(rc_names, "S", max_principal)
        matrix_all = per_angle(matrix, "S", max_principal)
        hoop = lambda e: e[1] * REBAR_E          # local axis 2 = hoop (cylindrical orientation)
        rebar_in = per_angle(band_inner, "E", hoop) if "E" in frame.fieldOutputs.keys() else {}
        rebar_out = per_angle(band_outer, "E", hoop) if "E" in frame.fieldOutputs.keys() else {}
        conc_rows = []
        for a in sorted(envelope_all):
            ri, ro = rebar_in.get(a, [float("nan")]), rebar_out.get(a, [float("nan")])
            conc_rows.append((a, max(envelope_all[a]) / 1e6,
                              max(matrix_all.get(a, [float("nan")])) / 1e6,
                              float(np.mean(ri)) / 1e6, float(np.mean(ro)) / 1e6))
        write_csv(os.path.join(args.out, step + "_concrete.csv"),
                  ("local_angle_deg", "max_principal_mpa", "matrix_max_principal_mpa",
                   "rebar_inner_stress_mpa", "rebar_outer_stress_mpa"), conc_rows)

        # ---- rock plasticity relative to installation
        plastic = dict(area_mean_m2=None, area_any_ip_m2=None, outer_radius_m=None,
                       reaches_outer_ring=None, max_norm=None)
        if "PE" in frame.fieldOutputs.keys():
            area_mean = area_any = 0.0
            rmax, nmax, outer_hit = 0.0, 0.0, False
            cell_rows = []
            for name, inst in (("ROCKIN-1", rin), ("ROCKOUT-1", rout)):
                pe = plastic_by_element(frame, inst)
                for e, ips in pe.items():
                    b = pe_base[name].get(e)
                    d = ips - (b if b is not None else 0.0)
                    nm = float(tensor_norm(d.mean(axis=0))[0])
                    ni = float(np.max(tensor_norm(d)))
                    cx, cy, a = rock_geo[name][e]
                    if ni > PLASTIC_THRESHOLD:
                        area_any += a
                    if nm > PLASTIC_THRESHOLD:
                        area_mean += a
                        rmax = max(rmax, math.hypot(cx, cy))
                        outer_hit = outer_hit or name == "ROCKOUT-1"
                        cell_rows.append((name, int(e), cx, cy, a, nm, ni))
                    nmax = max(nmax, nm)
            write_csv(os.path.join(args.out, step + "_rock_plastic_cells.csv"),
                      ("instance", "element", "x_m", "y_m", "area_m2", "dpe_norm_elem_mean",
                       "dpe_norm_ip_max"), cell_rows)
            plastic = dict(area_mean_m2=area_mean, area_any_ip_m2=area_any,
                           outer_radius_m=rmax, reaches_outer_ring=outer_hit, max_norm=nmax)

        i = int(np.argmax(envelope))
        row = dict(
            step=step, pressure_mpa=p_mpa,
            mean_contact_pressure_resultant_mpa=fr_sum / contact_len / 1e6,
            mean_contact_pressure_rock_stress_mpa=arc_mean(phi, q_rock, contact_lo, contact_hi) / 1e6,
            mean_contact_pressure_lining_stress_mpa=arc_mean(phi, q_lin, contact_lo, contact_hi) / 1e6,
            mean_contact_ur_mm=arc_mean(phi, ur_l, contact_lo, contact_hi) * 1e3,
            mean_void_ur_mm=(arc_mean(phi, ur_l, -half, half) * 1e3) if beta > 0 else None,
            seal_mises_max_mpa=float(envelope[i] / 1e6),
            seal_mises_max_local_angle_deg=float(phi[i]),
            seal_mises_max_surface="inner" if vm_env["seal_inner"][i] >= vm_env["seal_outer"][i] else "outer",
            seal_surface_peaks=surface_peak,
            seal_mises_integration_point_max_mpa=seal_ip_max,
            concrete_max_principal_max_mpa=max(r[1] for r in conc_rows),
            concrete_matrix_max_principal_max_mpa=float(np.nanmax([r[2] for r in conc_rows])),
            rebar_inner_stress_max_mpa=float(np.nanmax([r[3] for r in conc_rows])),
            rebar_outer_stress_max_mpa=float(np.nanmax([r[4] for r in conc_rows])),
            rebar_outer_stress_max_local_angle_deg=float(
                conc_rows[int(np.nanargmax([r[4] for r in conc_rows]))][0]),
            tie_displacement_gap_mm=gap * 1e3,
            rock_operational_plastic=plastic,
        )
        summaries.append(row)
        print(json.dumps(dict((k, v) for k, v in row.items() if k != "seal_surface_peaks")))

    result = dict(job=meta["job"], odb=os.path.abspath(args.odb), void_angle_deg=beta,
                  rock_model=meta["rock_model"], mesh=meta["mesh"],
                  excavation=dict(plastic_area_m2=exc_area, plastic_outer_radius_m=exc_rmax),
                  plastic_threshold=PLASTIC_THRESHOLD, steps=summaries)
    with open(os.path.join(args.out, "summary.json"), "w") as h:
        json.dump(result, h, indent=1)
    flat = []
    for r in summaries:
        p = r["rock_operational_plastic"]
        flat.append((r["step"], r["pressure_mpa"], r["mean_contact_pressure_resultant_mpa"],
                     r["mean_contact_pressure_rock_stress_mpa"],
                     r["mean_contact_pressure_lining_stress_mpa"], r["mean_contact_ur_mm"],
                     r["mean_void_ur_mm"] if r["mean_void_ur_mm"] is not None else "",
                     r["seal_mises_max_mpa"], r["seal_mises_max_local_angle_deg"],
                     r["seal_mises_integration_point_max_mpa"],
                     r["concrete_max_principal_max_mpa"],
                     r["concrete_matrix_max_principal_max_mpa"], r["rebar_inner_stress_max_mpa"],
                     r["rebar_outer_stress_max_mpa"],
                     p["area_mean_m2"] if p["area_mean_m2"] is not None else "",
                     p["outer_radius_m"] if p["outer_radius_m"] is not None else "",
                     r["tie_displacement_gap_mm"]))
    write_csv(os.path.join(args.out, "summary.csv"),
              ("step", "pressure_mpa", "q_mean_resultant_mpa", "q_mean_rock_stress_mpa",
               "q_mean_lining_stress_mpa", "contact_ur_mm", "void_ur_mm", "seal_mises_max_mpa",
               "seal_peak_local_angle_deg", "seal_mises_ip_max_mpa", "concrete_maxp_max_mpa",
               "matrix_maxp_max_mpa", "rebar_inner_max_mpa", "rebar_outer_max_mpa",
               "plastic_area_m2", "plastic_outer_radius_m", "tie_gap_mm"), flat)
    odb.close()
    print("written", os.path.abspath(args.out))


# Called unconditionally so that Abaqus/CAE "Run Script" (execfile) also runs it.
main()
