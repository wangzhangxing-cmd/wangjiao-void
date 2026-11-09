"""Finite-thickness modal operator for the seal and cracked RC lining.

The concrete lining is divided radially into a cracked matrix, an inner hoop-
reinforcement band and an outer hoop-reinforcement band.  Band locations and
widths are derived from the actual reinforcement.  The equivalent band width
preserves both steel area and the circular bar's centroidal second moment, so
the layered continuum reproduces the section A, B and D stiffnesses.

Fourier convention: u_r=Re(U exp(i*n*theta)),
u_theta=Re(T exp(i*n*theta)).  Boundary force vectors are external tractions on
each annulus face.  Plane strain in the cavern-axis direction is assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
from scipy.linalg import expm

from section_properties import SectionInput


@dataclass(frozen=True)
class OrthotropicMaterial:
    name: str
    reference_modulus_pa: float
    poisson_ratio: float
    hoop_ratio: float
    shear_ratio: float

    def plane_strain_stiffness(self) -> tuple[float, ...]:
        E, nu, eta, eta_g = (
            self.reference_modulus_pa,
            self.poisson_ratio,
            self.hoop_ratio,
            self.shear_ratio,
        )
        if not (E > 0.0 and -1.0 < nu < 0.5 and 0.0 < eta <= 1.0 and 0.0 < eta_g <= 1.0):
            raise ValueError("Invalid orthotropic engineering constants")
        s = math.sqrt(eta)
        compliance = np.array([
            [1.0 / E, -nu * s / E, -nu / E],
            [-nu * s / E, 1.0 / (eta * E), -nu * s / E],
            [-nu / E, -nu * s / E, 1.0 / E],
        ])
        if np.min(np.linalg.eigvalsh(compliance)) <= 0.0:
            raise ValueError("Orthotropic compliance is not positive definite")
        C = np.linalg.inv(compliance)
        G = eta_g * E / (2.0 * (1.0 + nu))
        return (
            float(C[0, 0]), float(C[0, 1]), float(C[1, 1]),
            float(C[0, 2]), float(C[1, 2]), float(G),
        )


@dataclass(frozen=True)
class RadialSegment:
    inner_radius_m: float
    outer_radius_m: float
    material: OrthotropicMaterial


@dataclass(frozen=True)
class LiningInput:
    inner_radius_m: float = 6.0
    outer_radius_m: float = 6.5
    concrete_modulus_pa: float = 33.4e9
    concrete_poisson: float = 0.20
    steel_modulus_pa: float = 200.0e9
    residual_hoop_ratio: float = 0.05
    residual_shear_ratio: float = 0.05
    cover_m: float = 0.050
    inner_bar_diameter_m: float = 0.012
    inner_bar_spacing_m: float = 0.200
    outer_bar_diameter_m: float = 0.020
    outer_bar_spacing_m: float = 0.200
    include_reinforcement: bool = True

    def section_input(self) -> SectionInput:
        return SectionInput(
            thickness_m=self.outer_radius_m - self.inner_radius_m,
            cover_m=self.cover_m,
            concrete_modulus_pa=self.concrete_modulus_pa,
            steel_modulus_pa=self.steel_modulus_pa,
            inner_diameter_m=self.inner_bar_diameter_m,
            inner_spacing_m=self.inner_bar_spacing_m,
            outer_diameter_m=self.outer_bar_diameter_m,
            outer_spacing_m=self.outer_bar_spacing_m,
        )

    def segments(self) -> tuple[RadialSegment, ...]:
        if not (0.0 < self.residual_hoop_ratio <= 1.0 and 0.0 < self.residual_shear_ratio <= 1.0):
            raise ValueError("The continuum operator requires positive residual ratios")
        section = self.section_input()
        bars = section.bar_layers() if self.include_reinforcement else ()
        definitions = []
        for bar in bars:
            centre = self.inner_radius_m + bar.centroid_from_inner_face_m
            width = bar.equal_second_moment_band_thickness_m
            definitions.append((bar, centre, width))
        boundaries = [self.inner_radius_m, self.outer_radius_m]
        for unused, centre, width in definitions:
            boundaries.extend([centre - width / 2.0, centre + width / 2.0])
        boundaries = sorted(set(boundaries))
        segments = []
        for ri, ro in zip(boundaries[:-1], boundaries[1:]):
            mid = (ri + ro) / 2.0
            hoop_ratio = self.residual_hoop_ratio
            labels = []
            for bar, centre, width in definitions:
                if abs(mid - centre) < width / 2.0 + 1.0e-12:
                    # The band replaces residual concrete over the steel area.
                    hoop_ratio += (
                        (self.steel_modulus_pa - self.residual_hoop_ratio * self.concrete_modulus_pa)
                        * bar.area_per_m
                        / (self.concrete_modulus_pa * width)
                    )
                    labels.append(bar.name + "_reinforcement")
            if hoop_ratio > 1.0 + 1.0e-12:
                raise ValueError("Equivalent hoop modulus exceeds the intact concrete modulus")
            material = OrthotropicMaterial(
                "+".join(labels) if labels else "cracked_concrete_matrix",
                self.concrete_modulus_pa,
                self.concrete_poisson,
                min(hoop_ratio, 1.0),
                self.residual_shear_ratio,
            )
            segments.append(RadialSegment(ri, ro, material))
        return tuple(segments)


def state_matrix(material: OrthotropicMaterial, order: int) -> np.ndarray:
    C11, C12, C22, unused_C13, unused_C23, C66 = material.plane_strain_stiffness()
    E = material.reference_modulus_pa
    a = C12 / C11
    b = (C22 - C12 * C12 / C11) / E
    c = E / C11
    d = E / C66
    n = int(order)
    return np.array([
        [-a, -n * a, c, 0.0],
        [n, 1.0, 0.0, d],
        [b, n * b, a, -n],
        [n * b, n * n * b, n * a, -1.0],
    ])


def _radial_n0(material: OrthotropicMaterial, ri: float, ro: float) -> np.ndarray:
    C11, C12, C22 = material.plane_strain_stiffness()[:3]
    k = math.sqrt(C22 / C11)
    D = np.array([[ri ** k, ri ** -k], [ro ** k, ro ** -k]])
    P, Q = C11 * k + C12, C12 - C11 * k
    F = np.array([
        [-P * ri ** (k - 1.0), -Q * ri ** (-k - 1.0)],
        [P * ro ** (k - 1.0), Q * ro ** (-k - 1.0)],
    ])
    return np.linalg.solve(D.T, F.T).T


def _torsion_n0(material: OrthotropicMaterial, ri: float, ro: float) -> np.ndarray:
    G = material.plane_strain_stiffness()[5]
    D = np.array([[ri, 1.0 / ri], [ro, 1.0 / ro]])
    F = np.array([[0.0, 2.0 * G / ri ** 2], [0.0, -2.0 * G / ro ** 2]])
    return np.linalg.solve(D.T, F.T).T


def single_annulus(material: OrthotropicMaterial, ri: float, ro: float, order: int) -> np.ndarray:
    if order == 0:
        S = np.zeros((4, 4), complex)
        S[np.ix_([0, 2], [0, 2])] = _radial_n0(material, ri, ro)
        S[np.ix_([1, 3], [1, 3])] = _torsion_n0(material, ri, ro)
        return S
    E = material.reference_modulus_pa
    H = expm(state_matrix(material, order) * math.log(ro / ri))
    h11, h12, h21, h22 = H[:2, :2], H[:2, 2:], H[2:, :2], H[2:, 2:]
    A = np.linalg.solve(h12, h11)
    B = np.linalg.solve(h12, np.eye(2))
    raw = E * np.block([
        [A / ri, -B / ri],
        [(h21 - h22 @ A) / ro, (h22 @ B) / ro],
    ])
    phase = np.array([1.0, -1j, 1.0, -1j])
    return phase[:, None] * raw / phase[None, :]


def _join(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    middle = left[2:, 2:] + right[:2, :2]
    B = np.hstack([left[2:, :2], right[:2, 2:]])
    C = np.vstack([left[:2, 2:], right[2:, :2]])
    return np.block([
        [left[:2, :2], np.zeros((2, 2))],
        [np.zeros((2, 2)), right[2:, 2:]],
    ]) - C @ np.linalg.solve(middle, B)


def _stable_pieces(segment: RadialSegment, order: int, exponent_limit: float) -> tuple[tuple[float, ...], tuple[np.ndarray, ...]]:
    if order == 0:
        count = 1
    else:
        eigenvalues = np.linalg.eigvals(state_matrix(segment.material, order))
        span = float(np.max(np.abs(eigenvalues.real))) * math.log(segment.outer_radius_m / segment.inner_radius_m)
        count = max(1, int(math.ceil(span / exponent_limit)))
    radii = tuple(np.geomspace(segment.inner_radius_m, segment.outer_radius_m, count + 1))
    pieces = tuple(
        single_annulus(segment.material, float(a), float(b), order)
        for a, b in zip(radii[:-1], radii[1:])
    )
    return radii, pieces


class RadiallyLayeredAnnulus:
    def __init__(self, segments: tuple[RadialSegment, ...], exponent_limit: float = 3.0):
        if not segments:
            raise ValueError("At least one radial segment is required")
        for left, right in zip(segments[:-1], segments[1:]):
            if abs(left.outer_radius_m - right.inner_radius_m) > 1.0e-12:
                raise ValueError("Radial segments must be contiguous")
        self.segments = tuple(segments)
        self.inner_radius_m = segments[0].inner_radius_m
        self.outer_radius_m = segments[-1].outer_radius_m
        self.exponent_limit = float(exponent_limit)
        self._cache = {}

    def chain(self, order: int):
        order = int(order)
        if order not in self._cache:
            radii = []
            pieces = []
            materials = []
            for segment in self.segments:
                rr, pp = _stable_pieces(segment, order, self.exponent_limit)
                if not radii:
                    radii.append(float(rr[0]))
                for j, piece in enumerate(pp):
                    radii.append(float(rr[j + 1]))
                    pieces.append(piece)
                    materials.append(segment.material)
            prefix = [pieces[0]]
            for right in pieces[1:]:
                prefix.append(_join(prefix[-1], right))
            self._cache[order] = (
                np.asarray(radii), tuple(pieces), tuple(prefix), tuple(materials)
            )
        return self._cache[order]

    def stiffness(self, order: int) -> np.ndarray:
        return self.chain(order)[2][-1].copy()

    def internal_displacements(self, order: int, inner_u: np.ndarray, outer_u: np.ndarray) -> np.ndarray:
        radii, pieces, prefix, unused_materials = self.chain(order)
        displacement = np.empty((len(pieces) + 1, 2), complex)
        displacement[0] = inner_u
        displacement[-1] = outer_u
        for k in range(len(pieces) - 1, 0, -1):
            left, right = prefix[k - 1], pieces[k]
            displacement[k] = np.linalg.solve(
                left[2:, 2:] + right[:2, :2],
                -left[2:, :2] @ inner_u - right[:2, 2:] @ displacement[k + 1],
            )
        return displacement

    def interface_force_gap(self, order: int, inner_u: np.ndarray, outer_u: np.ndarray) -> float:
        unused_radii, pieces, unused_prefix, unused_materials = self.chain(order)
        u = self.internal_displacements(order, inner_u, outer_u)
        gaps = []
        for k in range(len(pieces) - 1):
            left_outer = pieces[k][2:, :2] @ u[k] + pieces[k][2:, 2:] @ u[k + 1]
            right_inner = pieces[k + 1][:2, :2] @ u[k + 1] + pieces[k + 1][:2, 2:] @ u[k + 2]
            gaps.append(np.max(np.abs(left_outer + right_inner)))
        return float(max(gaps, default=0.0))


@dataclass(frozen=True)
class SealInput:
    inner_radius_m: float = 5.992
    outer_radius_m: float = 6.0
    elastic_modulus_pa: float = 210.0e9
    poisson_ratio: float = 0.30

    def annulus(self) -> RadiallyLayeredAnnulus:
        material = OrthotropicMaterial(
            "Q420_seal", self.elastic_modulus_pa, self.poisson_ratio, 1.0, 1.0
        )
        return RadiallyLayeredAnnulus((RadialSegment(self.inner_radius_m, self.outer_radius_m, material),))


class SealLiningOperator:
    """Maps outer displacement and inner-face loading to outer-face force."""

    def __init__(self, lining: LiningInput = LiningInput(), seal: SealInput = SealInput()):
        if abs(seal.outer_radius_m - lining.inner_radius_m) > 1.0e-12:
            raise ValueError("Seal and lining radii do not meet")
        self.lining_input = lining
        self.seal_input = seal
        self.seal = seal.annulus()
        self.lining = RadiallyLayeredAnnulus(lining.segments())

    @lru_cache(maxsize=4096)
    def mode(self, order: int) -> dict:
        S = self.seal.stiffness(order)
        C = self.lining.stiffness(order)
        A = np.block([
            [S[:2, :2], S[:2, 2:]],
            [S[2:, :2], S[2:, 2:] + C[:2, :2]],
        ])
        B = np.vstack([np.zeros((2, 2)), C[:2, 2:]])
        H = np.linalg.solve(A, -B)
        J = np.linalg.solve(A, np.vstack([np.eye(2), np.zeros((2, 2))]))
        K = C[2:, 2:] + C[2:, :2] @ H[2:]
        L = C[2:, :2] @ J[2:]
        Kraw, Lraw = K.copy(), L.copy()
        a = self.seal_input.inner_radius_m
        R = self.lining_input.outer_radius_m
        # Enforce exact rigid-rotation/translation identities after confirming
        # that the change removes roundoff only.
        if order == 0:
            K = np.diag([K[0, 0].real, 0.0]).astype(complex)
            L = np.diag([L[0, 0].real, -a * a / (R * R)]).astype(complex)
        elif order == 1:
            v = np.array([1.0, -1j])
            P = np.outer(v, v.conj()) / 2.0
            Q = np.eye(2) - P
            K = float(np.real(v.conj() @ K @ v)) / 4.0 * np.outer(v, v.conj())
            L = P @ L - a / R * Q
        Kgap = float(np.max(np.abs(K - Kraw)) / max(float(np.max(np.abs(Kraw))), 1.0))
        Lgap = float(np.max(np.abs(L - Lraw)) / max(float(np.max(np.abs(Lraw))), 1.0))
        if max(Kgap, Lgap) > 2.0e-10:
            raise FloatingPointError("Rigid-mode correction exceeds roundoff")
        return {
            "S": S, "C": C, "H": H, "J": J, "K": K, "L": L,
            "rigid_K_roundoff_relative": Kgap,
            "rigid_L_roundoff_relative": Lgap,
        }
