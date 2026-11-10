"""Generate Abaqus/Standard input files for the chapter-3 FE validation (v5).

Self-contained; runs with Python 2.7 or 3 and numpy, i.e. any of::

    abaqus python build_fem_model.py --matrix          # recommended set
    abaqus python build_fem_model.py --beta 10 --rock ep --mesh medium
    python build_fem_model.py --matrix
    Abaqus/CAE: File > Run Script (no arguments -> the recommended set)

Without arguments the six recommended models are written to ``models/``
next to this script (or the current directory if the script path is unknown).

Each call writes ``<job>.inp`` and ``<job>_meta.json`` into ``--out``.  The
meta file lists the surface node labels/angles that
``extract_fem_results.py`` needs, so the extractor never guesses geometry.

Model (plane strain, unit thickness, compressive pressure positive):

* LINER part: steel seal (``--seal-layers`` elements through 8 mm) and the
  five-layer cracked RC lining.  Seal and lining SHARE nodes (perfect bond).
  RC layers use the same engineering constants as the semi-analytical
  ``layered_ring_operator`` (cylindrical orientation: 1=r, 2=theta, 3=z).
* ROCKIN part: rock annulus R=6.5 m .. 3R, circumferential nodes identical to
  the liner (node-to-surface tie on the contact arc only, no tie on the void).
* ROCKOUT part: coarse rock annulus 3R .. R_out (default 40R), tied to ROCKIN
  with a surface-to-surface tie.  Plasticity must not reach it (checked by the
  extractor).  Far field: pressure p0 on the outer circle and three
  rigid-body supports.
* All elements CPE8R (quadratic, reduced integration).  The circumferential
  mesh is graded toward the two void endpoints, which coincide with nodes.

History (same as the semi-analytical model):

1. GEO       initial stress -p0 (all three normal components), wall pressure
             p0 and far-field pressure p0; liner removed.
2. EXCAVATE  wall support pressure ramped p0 -> 0 (``--excavation-increments``
             equal increments), i.e. the same p_s path as the semi-analytical
             excavation.  This replaces element removal, so no core mesh is
             needed and the unloading path is exactly prescribed.
3. ADD-LINING liner reactivated strain-free (displacement reference state).
4. P010 ...  internal gas pressure on the seal inner face, one step per
             requested pressure level (default 1, 2, ..., 10 MPa).
"""
from __future__ import division, print_function

import argparse
import io
import json
import math
import os
import sys

import numpy as np

try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # Abaqus/CAE "Run Script" may not set __file__
    HERE = os.getcwd()


MESH_PRESETS = {
    # theta sizes in degrees, radial sizes in metres
    # theta_* : lining/seal grid (graded to the void endpoints, resolves the seal peak)
    # rock_theta_min : rock-wall grid at the endpoints.  Kept coarser than the
    #   lining because the bonded/free corner is a stress singularity: refining the
    #   rock there concentrates plastic strain in a few tiny elements and stalls
    #   the Mohr-Coulomb return mapping (observed in V5_B10_EP_medium, 2026-11-10).
    "coarse": dict(theta_min=0.10, theta_max=1.00, theta_growth=1.15, rock_theta_min=0.50, seal_layers=2,
                   lining_dr=0.050, rock_dr0=0.020, rock_growth=1.10,
                   outer_theta_cells=360, outer_aspect=4.0),
    "medium": dict(theta_min=0.04, theta_max=0.50, theta_growth=1.10, rock_theta_min=0.25, seal_layers=4,
                   lining_dr=0.025, rock_dr0=0.010, rock_growth=1.08,
                   outer_theta_cells=720, outer_aspect=4.0),
    "fine": dict(theta_min=0.015, theta_max=0.25, theta_growth=1.07, rock_theta_min=0.125, seal_layers=6,
                 lining_dr=0.0125, rock_dr0=0.005, rock_growth=1.06,
                 outer_theta_cells=1440, outer_aspect=4.0),
}

# Lining / seal data identical to 02_code/core (layered_ring_operator.LiningInput
# and SealInput defaults with residual ratios 0.05).
LINING = dict(r_in=6.0, r_out=6.5, Ec=33.4e9, nu_c=0.20, Es=200.0e9, alpha=0.05,
              alpha_g=0.05, cover=0.050, bars=((0.012, 0.200), (0.020, 0.200)))
SEAL = dict(r_in=5.992, r_out=6.0)


class Material(object):
    def __init__(self, name, E, nu, hoop_ratio, shear_ratio):
        self.name, self.reference_modulus_pa, self.poisson_ratio = name, E, nu
        self.hoop_ratio, self.shear_ratio = hoop_ratio, shear_ratio


class Segment(object):
    def __init__(self, r_in, r_out, material):
        self.inner_radius_m, self.outer_radius_m, self.material = r_in, r_out, material


def lining_segments():
    """Cracked matrix plus two equal-second-moment hoop-reinforcement bands.

    Same algorithm as layered_ring_operator.LiningInput.segments()."""
    L = LINING
    t = L["r_out"] - L["r_in"]
    (d_in, s_in), (d_out, s_out) = L["bars"]
    bars = [("inner", d_in, s_in, L["cover"] + d_in / 2.0),
            ("outer", d_out, s_out, t - L["cover"] - d_out / 2.0)]
    defs = []
    for name, d, sp, y in bars:
        area = math.pi * d ** 2 / (4.0 * sp)
        defs.append((name, area, L["r_in"] + y, math.sqrt(3.0 / 4.0) * d))
    bounds = [L["r_in"], L["r_out"]]
    for _, _, centre, width in defs:
        bounds += [centre - width / 2.0, centre + width / 2.0]
    bounds = sorted(set(bounds))
    segments = []
    for ri, ro in zip(bounds[:-1], bounds[1:]):
        mid = 0.5 * (ri + ro)
        hoop, labels = L["alpha"], []
        for name, area, centre, width in defs:
            if abs(mid - centre) < width / 2.0 + 1.0e-12:
                hoop += (L["Es"] - L["alpha"] * L["Ec"]) * area / (L["Ec"] * width)
                labels.append(name + "_reinforcement")
        segments.append(Segment(ri, ro, Material(
            "+".join(labels) if labels else "cracked_concrete_matrix",
            L["Ec"], L["nu_c"], min(hoop, 1.0), L["alpha_g"])))
    return segments


ROCK = dict(E=12.0e9, nu=0.26, cohesion=0.9e6, phi=40.0, psi=10.0, p0=3.75e6)
STEEL = dict(E=210.0e9, nu=0.30)


# --------------------------------------------------------------------------
# 1-D grids
# --------------------------------------------------------------------------
def graded_segment(length, h_min, h_max, growth):
    """Element sizes over ``length``: h_min at the start, growing to h_max."""
    if length <= 0.0:
        return np.zeros(0)
    sizes, total, h = [], 0.0, h_min
    while total < length - 1.0e-12:
        sizes.append(h)
        total += h
        h = min(h_max, h * growth)
    sizes = np.asarray(sizes)
    # Drop a last cell that would be squeezed to less than half its neighbour.
    if len(sizes) > 2 and total - length > 0.5 * sizes[-1]:
        sizes = sizes[:-1]
    if len(sizes) == 1 or sizes[0] >= length:
        return np.array([length])
    # Keep the endpoint cell at h_min; absorb the mismatch in the other cells.
    rest = sizes[1:] * (length - sizes[0]) / sizes[1:].sum()
    return np.r_[sizes[0], rest]


def theta_breaks(beta_deg, mesh, h_min=None):
    """Corner angles in the local system phi (0 = crown), ascending in [-180,180).

    Void endpoints at +-beta/2 are nodes; the mesh is symmetric about the crown.
    """
    hmin, hmax, g = mesh["theta_min"], mesh["theta_max"], mesh["theta_growth"]
    if h_min is not None:
        hmin = min(h_min, hmax)
    if beta_deg <= 0.0:
        n = int(math.ceil(360.0 / hmax / 4.0)) * 4
        return -180.0 + np.arange(n) * 360.0 / n
    half = beta_deg / 2.0
    inside = graded_segment(half, hmin, hmax, g)[::-1]        # fine at +half
    outside = graded_segment(180.0 - half, hmin, hmax, g)      # fine at +half
    positive = np.r_[0.0, np.cumsum(np.r_[inside, outside])]   # 0 .. 180
    positive[-1] = 180.0
    phi = np.r_[-positive[::-1][:-1], positive[1:-1]]          # -180 .. <180
    phi = np.unique(np.round(phi, 12))
    if not (np.any(np.isclose(phi, half)) and np.any(np.isclose(phi, -half))):
        raise AssertionError("Void endpoints are not mesh nodes")
    return phi


def liner_radii(mesh):
    """Corner radii from the seal inner face to 6.5 m and the layer material of
    each radial element row."""
    radii = list(np.linspace(SEAL["r_in"], SEAL["r_out"], mesh["seal_layers"] + 1))
    rows = [("STEEL", -1)] * mesh["seal_layers"]
    for k, segment in enumerate(lining_segments()):
        n = max(1, int(math.ceil((segment.outer_radius_m - segment.inner_radius_m) / mesh["lining_dr"])))
        radii.extend(np.linspace(segment.inner_radius_m, segment.outer_radius_m, n + 1)[1:])
        rows.extend([("RC%d" % (k + 1), k)] * n)
    return np.asarray(radii), rows


def rock_inner_radii(R, r_tie, mesh):
    sizes = graded_segment(r_tie - R, mesh["rock_dr0"], 1.0e9, mesh["rock_growth"])
    radii = R + np.r_[0.0, np.cumsum(sizes)]
    radii[-1] = r_tie
    return radii


def rock_outer_radii(r_tie, r_out, n_theta, aspect):
    ratio = 1.0 + aspect * 2.0 * math.pi / n_theta
    n = max(4, int(math.ceil(math.log(r_out / r_tie) / math.log(ratio))))
    return r_tie * (r_out / r_tie) ** (np.arange(n + 1) / float(n))


# --------------------------------------------------------------------------
# Structured CPE8R annulus
# --------------------------------------------------------------------------
class Annulus(object):
    """Quadratic structured annulus. Node grid (a, b): a=0..2*nr, b=0..2*nt-1."""

    def __init__(self, radii, phi_deg):
        self.r = np.asarray(radii, float)
        self.phi = np.asarray(phi_deg, float)
        self.nr, self.nt = len(self.r) - 1, len(self.phi)
        phi_next = np.r_[self.phi[1:], self.phi[0] + 360.0]
        r_full = np.empty(2 * self.nr + 1)
        r_full[0::2] = self.r
        r_full[1::2] = 0.5 * (self.r[:-1] + self.r[1:])
        p_full = np.empty(2 * self.nt)
        p_full[0::2] = self.phi
        p_full[1::2] = 0.5 * (self.phi + phi_next)
        self.r_full, self.phi_full = r_full, p_full
        self.width = 2 * self.nt

    def label(self, a, b):
        return a * self.width + (b % self.width) + 1

    def nodes(self):
        for a, rr in enumerate(self.r_full):
            for b, pp in enumerate(self.phi_full):
                if a % 2 == 1 and b % 2 == 1:
                    continue
                t = math.radians(90.0 + pp)
                yield self.label(a, b), rr * math.cos(t), rr * math.sin(t)

    def element_label(self, i, j):
        return i * self.nt + j + 1

    def connectivity(self, i, j):
        a, b = 2 * i, 2 * j
        L = self.label
        return (L(a, b), L(a + 2, b), L(a + 2, b + 2), L(a, b + 2),
                L(a + 1, b), L(a + 2, b + 1), L(a + 1, b + 2), L(a, b + 1))

    def elements(self):
        for i in range(self.nr):
            for j in range(self.nt):
                yield self.element_label(i, j), self.connectivity(i, j)

    def element_mid_phi(self, j):
        return self.phi_full[2 * j + 1]

    def ring_nodes(self, i_corner):
        """Labels and phi of every node (corner + midside) on corner radius i."""
        a = 2 * i_corner
        return [self.label(a, b) for b in range(self.width)], list(self.phi_full)

    def check_orientation(self, coords):
        worst = np.inf
        for _, conn in self.elements():
            xy = np.array([coords[n] for n in (conn[0], conn[4], conn[1], conn[5],
                                               conn[2], conn[6], conn[3], conn[7])])
            x, y = xy[:, 0], xy[:, 1]
            area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
            worst = min(worst, area)
        if worst <= 0.0:
            raise AssertionError("Non-positive element area / clockwise element")
        return worst


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------
def fmt_list(values, per_line=16):
    values = list(values)
    return "\n".join(", ".join(str(v) for v in values[k:k + per_line])
                     for k in range(0, len(values), per_line))


def rc_constants(material):
    """Abaqus ENGINEERING CONSTANTS for the semi-analytical orthotropic law.

    Local axes (cylindrical orientation): 1=r, 2=theta, 3=z.  Reproduces the
    symmetric compliance of Eq. (6): S12=S23=-nu*sqrt(eta)/E, S13=-nu/E.
    """
    E, nu, eta, eta_g = (material.reference_modulus_pa, material.poisson_ratio,
                         material.hoop_ratio, material.shear_ratio)
    s = math.sqrt(eta)
    G = E / (2.0 * (1.0 + nu))
    return [E, eta * E, E, nu * s, nu, nu * s * eta, eta_g * G, G, eta_g * G]


def write_part(out, name, ann, elsets, surfaces, nsets, extra=""):
    out.append("*Part, name=%s" % name)
    out.append("*Node")
    for lab, x, y in ann.nodes():
        out.append("%d, %.12e, %.12e" % (lab, x, y))
    out.append("*Element, type=CPE8R")
    for lab, conn in ann.elements():
        out.append("%d, %s" % (lab, ", ".join(str(c) for c in conn)))
    out.append("*Elset, elset=ALL, generate\n1, %d, 1" % (ann.nr * ann.nt))
    out.append("*Nset, nset=ALL, elset=ALL")
    for set_name, labels in elsets.items():
        out.append("*Elset, elset=%s\n%s" % (set_name, fmt_list(labels)))
    for set_name, labels in nsets.items():
        out.append("*Nset, nset=%s\n%s" % (set_name, fmt_list(labels)))
    for surf, (elset, face) in surfaces.items():
        out.append("*Surface, type=ELEMENT, name=%s\n%s, %s" % (surf, elset, face))
    if extra:
        out.append(extra)
    out.append("*End Part")


def build(beta_deg, rock_model, mesh_name, pressures_mpa,
          out_dir, r_out_factor, excavation_increments,
          job=None):
    mesh = MESH_PRESETS[mesh_name]
    R = LINING["r_out"]
    r_tie, r_out = 3.0 * R, r_out_factor * R
    half = beta_deg / 2.0
    job = job or "V5_B%02d_%s_%s" % (int(round(beta_deg)), rock_model.upper(), mesh_name)

    phi = theta_breaks(beta_deg, mesh)
    phi_rock = theta_breaks(beta_deg, mesh, mesh["rock_theta_min"])
    lr, rows = liner_radii(mesh)
    liner = Annulus(lr, phi)
    rin = Annulus(rock_inner_radii(R, r_tie, mesh), phi_rock)
    n_out = mesh["outer_theta_cells"]
    rout = Annulus(rock_outer_radii(r_tie, r_out, n_out, mesh["outer_aspect"]),
                   -180.0 + np.arange(n_out) * 360.0 / n_out)

    def in_void(p):
        return beta_deg > 0.0 and abs(p) < half - 1.0e-9

    # ---- liner sets
    segments = lining_segments()
    liner_elsets = {"SEAL": [], "RC": []}
    for k in range(len(segments)):
        liner_elsets["RC%d" % (k + 1)] = []
    for i, (mat, k) in enumerate(rows):
        labels = [liner.element_label(i, j) for j in range(liner.nt)]
        if mat == "STEEL":
            liner_elsets["SEAL"].extend(labels)
        else:
            liner_elsets["RC"].extend(labels)
            liner_elsets[mat].extend(labels)
    liner_elsets["SEAL_IN_ROW"] = [liner.element_label(0, j) for j in range(liner.nt)]
    liner_elsets["OUT_CONTACT_ROW"] = [liner.element_label(liner.nr - 1, j) for j in range(liner.nt)
                                       if not in_void(liner.element_mid_phi(j))]
    liner_elsets["OUT_ROW"] = [liner.element_label(liner.nr - 1, j) for j in range(liner.nt)]
    seal_in_nodes, seal_phi = liner.ring_nodes(0)
    seal_out_nodes, _ = liner.ring_nodes(mesh["seal_layers"])
    liner_out_nodes, _ = liner.ring_nodes(liner.nr)
    liner_nsets = {"SEAL_INNER_N": seal_in_nodes, "SEAL_OUTER_N": seal_out_nodes,
                   "LINER_OUTER_N": liner_out_nodes}
    liner_surfaces = {"SEAL_INNER": ("SEAL_IN_ROW", "S4"),
                      "OUTER_CONTACT": ("OUT_CONTACT_ROW", "S2")}
    orientation = ("*Orientation, name=CYL, system=CYLINDRICAL\n"
                   "0., 0., 0., 0., 0., 1.\n3, 0.")
    sections = ["*Solid Section, elset=SEAL, material=STEEL\n1.,"]
    for k in range(len(segments)):
        sections.append("*Solid Section, elset=RC%d, material=RC%d, orientation=CYL\n1.,"
                        % (k + 1, k + 1))

    # ---- rock sets
    rin_elsets = {
        "WALL_ROW": [rin.element_label(0, j) for j in range(rin.nt)],
        "WALL_CONTACT_ROW": [rin.element_label(0, j) for j in range(rin.nt)
                             if not in_void(rin.element_mid_phi(j))],
        "TIE_ROW": [rin.element_label(rin.nr - 1, j) for j in range(rin.nt)],
    }
    wall_nodes, wall_phi = rin.ring_nodes(0)
    rin_nsets = {"WALL_N": wall_nodes}
    rin_surfaces = {"WALL": ("WALL_ROW", "S4"), "WALL_CONTACT": ("WALL_CONTACT_ROW", "S4"),
                    "TIE_OUT": ("TIE_ROW", "S2")}
    rout_elsets = {"TIE_ROW": [rout.element_label(0, j) for j in range(rout.nt)],
                   "OUTER_ROW": [rout.element_label(rout.nr - 1, j) for j in range(rout.nt)]}
    a_out = 2 * rout.nr
    q = rout.width // 4
    # phi grid starts at -180 deg (global -90 deg): global 0/90/180 deg nodes
    support = {"FIX_Y": [rout.label(a_out, q), rout.label(a_out, 3 * q)],
               "FIX_X": [rout.label(a_out, 2 * q)]}
    rout_nsets = dict(support)
    rout_surfaces = {"TIE_IN": ("TIE_ROW", "S4"), "OUTER": ("OUTER_ROW", "S2")}
    rock_section = "*Solid Section, elset=ALL, material=ROCK\n1.,"

    # ---- geometry checks
    coords = {}
    for ann in (liner, rin, rout):
        coords.clear()
        coords.update({lab: (x, y) for lab, x, y in ann.nodes()})
        ann.check_orientation(coords)
    lc = {lab: (x, y) for lab, x, y in liner.nodes()}
    rc = {lab: (x, y) for lab, x, y in rin.nodes()}
    # Both grids contain the void endpoints and lie on r = 6.5 m (non-matching tie).
    for nodes, ann, ring in ((liner_out_nodes, lc, "liner"), (wall_nodes, rc, "rock")):
        radius = max(abs(math.hypot(*ann[n]) - R) for n in nodes)
        if radius > 1.0e-9:
            raise AssertionError("%s interface nodes are not on r = R" % ring)
    if beta_deg > 0.0:
        for grid in (phi, phi_rock):
            if not (np.any(np.isclose(grid, half)) and np.any(np.isclose(grid, -half))):
                raise AssertionError("Void endpoint missing from an interface grid")
    oc = {lab: (x, y) for lab, x, y in rout.nodes()}
    for name, idx in (("FIX_Y", 0), ("FIX_Y", 1), ("FIX_X", 0)):
        x, y = oc[support[name][idx]]
        if name == "FIX_Y" and abs(y) > 1.0e-6 * r_out or name == "FIX_X" and abs(x) > 1.0e-6 * r_out:
            raise AssertionError("Support node is not on an axis")

    # ---- input file
    o = []
    o.append("*Heading\n** %s: chapter-3 FE validation v5, void %.4g deg at crown, rock=%s, mesh=%s"
             % (job, beta_deg, rock_model, mesh_name))
    o.append("** generated by 03_fem/fem_v5/build_fem_model.py; units N, m, Pa")
    o.append("*Preprint, echo=NO, model=NO, history=NO, contact=NO")
    write_part(o, "LINER", liner, liner_elsets, liner_surfaces, liner_nsets,
               orientation + "\n" + "\n".join(sections))
    write_part(o, "ROCKIN", rin, rin_elsets, rin_surfaces, rin_nsets, rock_section)
    write_part(o, "ROCKOUT", rout, rout_elsets, rout_surfaces, rout_nsets, rock_section)
    o.append("*Assembly, name=Assembly")
    for inst, part in (("LINER-1", "LINER"), ("ROCKIN-1", "ROCKIN"), ("ROCKOUT-1", "ROCKOUT")):
        o.append("*Instance, name=%s, part=%s\n*End Instance" % (inst, part))
    o.append("** Liner contact arc bonded to the rock wall; the void arc is free.")
    o.append("** Lining grid is finer than the rock grid: lining = slave (non-matching tie).")
    o.append("*Tie, name=LINER-ROCK, adjust=NO, type=NODE TO SURFACE\n"
             "LINER-1.OUTER_CONTACT, ROCKIN-1.WALL_CONTACT")
    o.append("*Tie, name=ROCK-RING, adjust=YES, type=SURFACE TO SURFACE\n"
             "ROCKIN-1.TIE_OUT, ROCKOUT-1.TIE_IN")
    o.append("*End Assembly")

    o.append("*Material, name=STEEL\n*Elastic\n%.6e, %.6g" % (STEEL["E"], STEEL["nu"]))
    for k, segment in enumerate(segments):
        c = rc_constants(segment.material)
        o.append("** RC%d: %s, r=%.6f..%.6f m, hoop ratio %.9g"
                 % (k + 1, segment.material.name, segment.inner_radius_m,
                    segment.outer_radius_m, segment.material.hoop_ratio))
        o.append("*Material, name=RC%d\n*Elastic, type=ENGINEERING CONSTANTS\n%s,\n%s,"
                 % (k + 1, ", ".join("%.12g" % v for v in c[:8]), "%.12g" % c[8]))
    o.append("*Material, name=ROCK\n*Elastic\n%.6e, %.6g" % (ROCK["E"], ROCK["nu"]))
    if rock_model == "ep":
        o.append("*Mohr Coulomb\n%.6g, %.6g\n*Mohr Coulomb Hardening\n%.6e, 0."
                 % (ROCK["phi"], ROCK["psi"], ROCK["cohesion"]))

    o.append("*Boundary\nROCKOUT-1.FIX_Y, 2, 2, 0.\nROCKOUT-1.FIX_X, 1, 1, 0.")
    p0 = ROCK["p0"]
    o.append("*Initial Conditions, type=STRESS\nROCKIN-1.ALL, %.6e, %.6e, %.6e, 0.\n"
             "ROCKOUT-1.ALL, %.6e, %.6e, %.6e, 0." % ((-p0,) * 6))

    unsymm = "YES" if rock_model == "ep" else "NO"
    # Solution controls of the converged V4 model: discontinuous analysis, 10%
    # displacement-correction tolerance, line search, more attempts/cutbacks.
    controls = ("*Controls, analysis=discontinuous\n"
                "*Controls, parameters=field, field=displacement\n, 0.1\n"
                "*Controls, parameters=line search\n5\n"
                "*Controls, parameters=time incrementation\n, , , , , , 20, 15, , ,\n"
                if rock_model == "ep" else "")
    rock_out = ("*Element Output, elset=ROCKIN-1.ALL, directions=YES\nPE, PEEQ\n"
                "*Element Output, elset=ROCKOUT-1.ALL, directions=YES\nPE, PEEQ\n"
                if rock_model == "ep" else "")
    # Restart data after every increment, latest one only (OVERLAY), so a job
    # terminated from outside can be continued with continue_job.py.
    restart = "*Restart, write, frequency=1, overlay\n" if rock_model == "ep" else ""
    field = ("*Output, field, frequency=10000\n*Node Output\nU, RF\n"
             "*Element Output, directions=YES\nS, E, NFORC\n" + rock_out +
             "*Output, history, variable=PRESELECT\n")

    o.append("*Step, name=GEO, nlgeom=NO, unsymm=%s\n*Geostatic\n"
             "*Model Change, remove\nLINER-1.ALL,\n"
             "*Dsload\nROCKIN-1.WALL, P, %.6e\nROCKOUT-1.OUTER, P, %.6e\n%s*End Step"
             % (unsymm, p0, p0, field))
    dt = 1.0 / excavation_increments
    o.append("*Step, name=EXCAVATE, nlgeom=NO, inc=10000, unsymm=%s\n*Static\n"
             "%.8g, 1., %.3g, %.8g\n%s"
             "** support pressure ramped linearly p0 -> 0 (default static ramp)\n"
             "*Dsload, op=MOD\nROCKIN-1.WALL, P, 0.\n%s%s*End Step"
             % (unsymm, dt, dt * 1.0e-4, dt, controls, restart, field))
    o.append("*Step, name=ADD-LINING, nlgeom=NO, unsymm=%s\n*Static\n1., 1., 1e-05, 1.\n"
             "*Model Change, add\nLINER-1.ALL,\n%s%s*End Step" % (unsymm, restart, field))
    for p in pressures_mpa:
        name = "P%03d" % int(round(10 * p))
        # elastic: one increment suffices; plastic: max 0.1 MPa per increment
        # (the semi-analytical history uses 0.1 MPa steps).
        inc_line = "0.05, 1., 1e-08, 0.1" if rock_model == "ep" else "0.5, 1., 1e-06, 1."
        o.append("*Step, name=%s, nlgeom=NO, inc=100000, unsymm=%s\n*Static\n"
                 "%s\n%s*Dsload, op=MOD\nLINER-1.SEAL_INNER, P, %.6e\n%s%s*End Step"
                 % (name, unsymm, inc_line, controls, p * 1.0e6, restart, field))

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    inp = os.path.join(out_dir, job + ".inp")
    with open(inp, "w") as handle:
        handle.write("\n".join(o) + "\n")

    meta = dict(
        job=job, version="fem-v5", void_angle_deg=beta_deg, rock_model=rock_model,
        mesh=mesh_name, mesh_parameters=mesh, pressures_mpa=list(pressures_mpa),
        pressure_steps=["P%03d" % int(round(10 * p)) for p in pressures_mpa],
        angle_convention="local phi = global theta - 90 deg; void centred on the crown",
        radii=dict(seal_inner=float(lr[0]), seal_outer=float(lr[mesh["seal_layers"]]),
                   lining_outer=R, rock_tie=r_tie, rock_outer=r_out),
        counts=dict(theta_cells=int(liner.nt), liner_rows=int(liner.nr),
                    rock_inner_rows=int(rin.nr), rock_outer_rows=int(rout.nr),
                    rock_outer_theta_cells=int(rout.nt),
                    elements=int(liner.nr * liner.nt + rin.nr * rin.nt + rout.nr * rout.nt)),
        min_theta_cell_deg=float(np.min(np.diff(np.r_[phi, phi[0] + 360]))),
        min_rock_theta_cell_deg=float(np.min(np.diff(np.r_[phi_rock, phi_rock[0] + 360]))),
        liner_layers=[dict(row=i, material=m) for i, (m, _) in enumerate(rows)],
        rc_layers=[dict(name="RC%d" % (k + 1), label=s.material.name,
                        r_in=s.inner_radius_m, r_out=s.outer_radius_m,
                        hoop_ratio=s.material.hoop_ratio) for k, s in enumerate(segments)],
        surface_nodes=dict(
            phi_deg=[float(v) for v in seal_phi],
            seal_inner=seal_in_nodes, seal_outer=seal_out_nodes,
            liner_outer=liner_out_nodes, rock_wall=wall_nodes,
            rock_phi_deg=[float(v) for v in wall_phi]),
        rock_inner_radii=[float(v) for v in rin.r],
        material=dict(rock=ROCK, steel=STEEL, abaqus_mohr_coulomb=(
            "Abaqus smooths the Mohr-Coulomb flow potential (meridional eccentricity 0.1, "
            "Menetrey-Willam deviatoric); the semi-analytical model uses the classical "
            "nonsmooth surfaces. Quantify by the beta=0 comparison.")),
    )
    with open(os.path.join(out_dir, job + "_meta.json"), "w") as handle:
        handle.write(json.dumps(meta, indent=1))
    print("%s: %d elements (theta cells %d, liner rows %d, rock rows %d+%d), "
          "min theta cell %.4f deg -> %s"
          % (job, meta["counts"]["elements"], liner.nt, liner.nr, rin.nr, rout.nr,
             meta["min_theta_cell_deg"], inp))
    return inp


def main():
    ap = argparse.ArgumentParser(description="Generate v5 Abaqus validation models")
    ap.add_argument("--beta", type=float, default=10.0, help="void angle in degrees (0 = no void)")
    ap.add_argument("--rock", choices=("el", "ep"), default="ep")
    ap.add_argument("--mesh", choices=tuple(MESH_PRESETS), default="medium")
    ap.add_argument("--pressures", default="1,2,3,4,5,6,7,8,9,10",
                    help="comma separated internal pressures in MPa (one step each)")
    ap.add_argument("--r-out", type=float, default=40.0, help="outer radius / 6.5 m")
    ap.add_argument("--excavation-increments", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(HERE, "models"))
    ap.add_argument("--matrix", action="store_true",
                    help="write the recommended validation set (see README)")
    argv = [v for v in sys.argv[1:] if v.startswith("--") or not v.startswith("-")]
    known = [v for v in argv if v.split("=")[0] in {o for act in ap._actions for o in act.option_strings}]
    a, _ = ap.parse_known_args(argv)
    # Abaqus/CAE Run Script passes no script arguments: build the recommended set.
    run_matrix = a.matrix or not known
    pressures = [float(v) for v in a.pressures.split(",") if v.strip()]
    out = a.out
    print("writing to", out)
    if run_matrix:
        for beta, rock, mesh in ((10, "el", "medium"), (10, "ep", "medium"),
                                 (0, "el", "medium"), (0, "ep", "medium"),
                                 (10, "ep", "coarse"), (10, "ep", "fine")):
            build(beta, rock, mesh, pressures, out, a.r_out, a.excavation_increments)
    else:
        build(a.beta, a.rock, a.mesh, pressures, out, a.r_out, a.excavation_increments)


# Called unconditionally so that Abaqus/CAE "Run Script" (execfile) also runs it.
main()
