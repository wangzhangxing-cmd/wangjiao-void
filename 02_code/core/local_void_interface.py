"""Elastic 30-degree free-sector interface for the new layered lining.

The contact arc is bonded in radial and tangential directions.  The prescribed
void arc carries zero traction.  Piecewise-constant contact tractions are used
only for the present interface verification; endpoint-local stresses are not
interpreted until the weighted endpoint basis is added.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from layered_ring_operator import LiningInput, SealInput, SealLiningOperator, state_matrix


def regular_disk_compliance(orders: np.ndarray, material, radius: float) -> np.ndarray:
    orders = np.atleast_1d(np.asarray(orders, int))
    out = np.zeros((len(orders), 2, 2), complex)
    E = material.reference_modulus_pa
    phase = np.array([1.0, -1j])
    for index, n in enumerate(orders):
        if n < 2:
            raise ValueError("Regular disk tail requires n>=2")
        eigenvalues, eigenvectors = np.linalg.eig(state_matrix(material, int(n)))
        keep = np.argsort(-eigenvalues.real)[:2]
        if np.min(eigenvalues.real[keep]) <= 0.0:
            raise FloatingPointError("Two regular disk modes were not found")
        displacement = eigenvectors[:2, keep]
        traction_state = eigenvectors[2:, keep]
        stiffness = (E / radius) * traction_state @ np.linalg.inv(displacement)
        stiffness = phase[:, None] * stiffness / phase[None, :]
        out[index] = np.linalg.inv(stiffness)
    return out


def rock_compliance(maximum_order: int, E: float = 12.0e9, nu: float = 0.26,
                    radius: float = 6.5) -> np.ndarray:
    G = E / (2.0 * (1.0 + nu))
    out = np.zeros((maximum_order + 1, 2, 2), complex)
    out[0] = radius / (2.0 * G) * np.eye(2)
    v = np.array([1.0, -1j])
    out[1] = radius / (8.0 * G) * np.outer(v, v.conj())
    for n in range(2, maximum_order + 1):
        diagonal = 2.0 * G * (2.0 * n + 1.0 - 2.0 * nu * (n + 1.0)) / (radius * (3.0 - 4.0 * nu))
        off = 2.0 * G * (n + 2.0 - 2.0 * nu * (n + 1.0)) / (radius * (3.0 - 4.0 * nu))
        out[n] = np.array([[diagonal, -1j * off], [1j * off, diagonal]]) / (diagonal ** 2 - off ** 2)
    return out


class LayerRockCompliance:
    def __init__(self, core_order: int = 1024, maximum_order: int = 8192,
                 rock_E: float = 12.0e9, rock_nu: float = 0.26,
                 lining_input: LiningInput | None = None,
                 seal_input: SealInput | None = None):
        if not 2 <= core_order <= maximum_order:
            raise ValueError("Invalid modal orders")
        self.core_order = int(core_order)
        self.order = int(maximum_order)
        if lining_input is None:
            lining_input = LiningInput(
                residual_hoop_ratio=0.05, residual_shear_ratio=0.05
            )
        if seal_input is None:
            seal_input = SealInput()
        self.layers = SealLiningOperator(lining_input, seal_input)
        self.radius = self.layers.lining_input.outer_radius_m
        self.rock_E, self.rock_nu = float(rock_E), float(rock_nu)
        self.CR = rock_compliance(self.order, self.rock_E, self.rock_nu, self.radius)
        self.CL = np.zeros_like(self.CR)
        self.K = np.zeros_like(self.CR)
        self.L0 = self.layers.mode(0)["L"].copy()
        for n in range(self.core_order + 1):
            K = self.layers.mode(n)["K"]
            self.K[n] = K
            if n == 0:
                self.CL[0, 0, 0] = 1.0 / K[0, 0]
            elif n == 1:
                v = np.array([1.0, -1j])
                P = np.outer(v, v.conj()) / 2.0
                eigenvalue = float(np.real(v.conj() @ K @ v) / 2.0)
                self.CL[1] = P / eigenvalue
            else:
                self.CL[n] = np.linalg.inv(K)
        outer_material = self.layers.lining.segments[-1].material
        if self.order > self.core_order:
            orders = np.arange(self.core_order + 1, self.order + 1)
            self.CL[self.core_order + 1:] = regular_disk_compliance(
                orders, outer_material, self.radius
            )
            self.K[self.core_order + 1:] = np.linalg.inv(self.CL[self.core_order + 1:])
        self.compliance = self.CL + self.CR
        hermitian_gap = np.max(
            np.abs(self.compliance - self.compliance.conj().transpose(0, 2, 1)), axis=(1, 2)
        ) / np.maximum(np.max(np.abs(self.compliance), axis=(1, 2)), 1.0e-300)
        self.maximum_reciprocity_gap = float(np.max(hermitian_gap))
        if self.maximum_reciprocity_gap > 1.0e-8:
            raise FloatingPointError("Combined compliance lost reciprocity")
        # Tail must approach the last exact finite-annulus modes smoothly.
        probes = np.arange(max(2, self.core_order - 31), self.core_order + 1)
        disk = regular_disk_compliance(probes, outer_material, self.radius)
        exact = self.CL[probes]
        self.disk_tail_probe_max_relative_gap = float(
            np.max(np.abs(disk - exact) / np.maximum(np.max(np.abs(exact), axis=(1, 2))[:, None, None], 1.0e-300))
        )


def traction_coefficients(edges: np.ndarray, panel_traction: np.ndarray,
                          order: int, chunk: int = 512) -> np.ndarray:
    width = float(edges[1] - edges[0])
    mid = (edges[:-1] + edges[1:]) / 2.0
    out = np.empty((order + 1, 2), complex)
    for first in range(0, order + 1, chunk):
        n = np.arange(first, min(first + chunk, order + 1))
        factor = width / np.pi * np.sinc(n * width / (2.0 * np.pi))
        if first == 0:
            factor[0] /= 2.0
        out[first:first + len(n)] = factor[:, None] * (
            np.exp(-1j * n[:, None] * mid[None]) @ panel_traction
        )
    return out


def arc_mean(coefficients: np.ndarray, start: float, end: float) -> np.ndarray:
    if end <= start:
        raise ValueError("Arc length must be positive")
    n = np.arange(len(coefficients))
    weights = np.sinc(n * (end - start) / (2.0 * np.pi)) * np.exp(1j * n * (start + end) / 2.0)
    return np.real(weights @ coefficients)


def panel_integrals(coefficients: np.ndarray, edges: np.ndarray,
                    chunk: int = 512) -> np.ndarray:
    """Integrate a real Fourier field over every interface panel."""
    coefficients = np.asarray(coefficients, complex)
    width = np.diff(edges)
    mid = (edges[:-1] + edges[1:]) / 2.0
    result = width[:, None] * coefficients[0].real
    for first in range(1, len(coefficients), chunk):
        n = np.arange(first, min(first + chunk, len(coefficients)))
        moment = (
            width[None, :] * np.sinc(n[:, None] * width[None, :] / (2.0 * np.pi))
            * np.exp(1j * n[:, None] * mid[None, :])
        )
        result += np.real(moment.T @ coefficients[first:first + len(n)])
    return result


class FreeSectorInterface:
    def __init__(self, void_angle_deg: float = 30.0, contact_panels: int = 256,
                 core_order: int = 1024, maximum_order: int = 8192,
                 rock_E: float = 12.0e9, rock_nu: float = 0.26,
                 lining_input: LiningInput | None = None,
                 seal_input: SealInput | None = None):
        self.void_angle_deg = float(void_angle_deg)
        self.panels = int(contact_panels)
        self.alpha = np.deg2rad(self.void_angle_deg / 2.0)
        self.op = LayerRockCompliance(
            core_order, maximum_order, rock_E=rock_E, rock_nu=rock_nu,
            lining_input=lining_input, seal_input=seal_input,
        )
        self.order = self.op.order
        self.edges = np.linspace(self.alpha, 2.0 * np.pi - self.alpha, self.panels + 1)
        self.mid = (self.edges[:-1] + self.edges[1:]) / 2.0
        self.width = float(self.edges[1] - self.edges[0])
        lag = np.arange(1 - self.panels, self.panels)
        rr = np.zeros(len(lag)); rt = np.zeros(len(lag)); tt = np.zeros(len(lag))
        for first in range(0, self.order + 1, 512):
            n = np.arange(first, min(first + 512, self.order + 1))
            factor = self.width / np.pi * np.sinc(n * self.width / (2.0 * np.pi))
            weights = np.full(len(n), np.pi)
            if first == 0:
                factor[0] /= 2.0
                weights[0] = 2.0 * np.pi
            phase = np.exp(1j * n[:, None] * lag[None] * self.width)
            weighted = weights * factor ** 2
            D = self.op.compliance[n]
            rr += np.real(np.einsum("n,nl->l", weighted * D[:, 0, 0], phase))
            rt += np.real(np.einsum("n,nl->l", weighted * D[:, 0, 1], phase))
            tt += np.real(np.einsum("n,nl->l", weighted * D[:, 1, 1], phase))
        ids = self.panels - 1 + np.arange(self.panels)[:, None] - np.arange(self.panels)[None]
        self.H = np.block([[rr[ids], rt[ids]], [rt[ids].T, tt[ids]]])
        c = 2.0 * np.sin(self.width / 2.0) * np.cos(self.mid)
        s = 2.0 * np.sin(self.width / 2.0) * np.sin(self.mid)
        self.Z = np.zeros((2 * self.panels, 3))
        self.Z[:self.panels, 0] = c
        self.Z[:self.panels, 1] = s
        self.Z[self.panels:, 0] = -s
        self.Z[self.panels:, 1] = c
        self.Z[self.panels:, 2] = self.width
        self.pressure_scale = 1.0e6
        self.displacement_scale = self.op.radius * self.pressure_scale / self.op.rock_E
        self.system = np.block([
            [self.H * self.pressure_scale / self.displacement_scale, -self.Z],
            [-self.Z.T, np.zeros((3, 3))],
        ])
        self.lu = lu_factor(self.system)

    def solve(self, internal_pressure_pa: float, free_rock_displacement=None) -> dict:
        inner = np.array([float(internal_pressure_pa), 0.0])
        h0 = -self.op.CL[0] @ self.op.L0 @ inner
        rhs = np.tile(self.width * h0.real, (self.panels, 1)).T.ravel()
        if free_rock_displacement is None:
            free_wall = np.zeros((1, 2), complex)
        else:
            free_wall = np.asarray(free_rock_displacement, complex)
            if (free_wall.ndim != 2 or free_wall.shape[1] != 2 or
                    len(free_wall) > self.order + 1 or not np.all(np.isfinite(free_wall))):
                raise ValueError("Invalid plastic free-wall displacement coefficients")
            scale = max(float(np.max(np.abs(free_wall))), 1.0e-30)
            if np.max(np.abs(free_wall[0].imag)) > 1.0e-12 * scale:
                raise ValueError("Mean free-wall displacement must be real")
        projection = panel_integrals(free_wall, self.edges)
        rhs -= projection.T.ravel()
        scaled = np.r_[rhs / self.displacement_scale, np.zeros(3)]
        solution = lu_solve(self.lu, scaled)
        residual = self.system @ solution - scaled
        q = solution[:-3].reshape(2, self.panels).T * self.pressure_scale
        rigid = solution[-3:] * self.displacement_scale
        traction = traction_coefficients(self.edges, q, self.order)
        lining = -np.einsum("nab,nb->na", self.op.CL, traction)
        lining[0] += h0
        lining[0, 1] += rigid[2]
        lining[1] += np.array([rigid[0] - 1j * rigid[1], 1j * rigid[0] + rigid[1]])
        rock = np.einsum("nab,nb->na", self.op.CR, traction)
        rock[:len(free_wall)] += free_wall
        weak = self.H @ (q.T.ravel()) - self.Z @ rigid - rhs
        balance = self.Z.T @ q.T.ravel()
        return {
            "panel_traction": q,
            "traction_coefficients": traction,
            "lining_u": lining,
            "rock_u": rock,
            "rigid_displacement": rigid,
            "linear_relative_residual": float(np.max(np.abs(residual)) / max(np.max(np.abs(scaled)), 1.0e-30)),
            "weak_displacement_gap_m": float(np.max(np.abs(weak)) / self.width),
            "force_torque_balance_gap_pa": float(np.max(np.abs(balance))),
        }

    def metrics(self, result: dict) -> dict:
        q = result["panel_traction"]
        endpoint_distance = np.minimum(
            self.mid - self.alpha, 2.0 * np.pi - self.alpha - self.mid
        )
        interior = endpoint_distance >= np.deg2rad(5.0)
        if not np.any(interior):
            raise FloatingPointError("No contact panels remain after endpoint exclusion")
        contact_u = arc_mean(result["lining_u"], self.alpha, 2.0 * np.pi - self.alpha)
        void_u = arc_mean(result["lining_u"], -self.alpha, self.alpha)
        return {
            "mean_contact_pressure_mpa": float(np.mean(q[:, 0]) / 1.0e6),
            "minimum_contact_pressure_mpa": float(np.min(q[:, 0]) / 1.0e6),
            "maximum_contact_pressure_mpa": float(np.max(q[:, 0]) / 1.0e6),
            "fraction_contact_panels_in_tension": float(np.mean(q[:, 0] < 0.0)),
            "maximum_contact_shear_mpa": float(np.max(np.abs(q[:, 1])) / 1.0e6),
            "endpoint_exclusion_deg": 5.0,
            "interior_radial_pressure_p99_mpa": float(np.quantile(q[interior, 0], 0.99) / 1.0e6),
            "interior_absolute_shear_p99_mpa": float(np.quantile(np.abs(q[interior, 1]), 0.99) / 1.0e6),
            "mean_contact_radial_displacement_mm": float(contact_u[0] * 1000.0),
            "mean_void_radial_displacement_mm": float(void_u[0] * 1000.0),
            "void_to_contact_displacement_ratio": float(void_u[0] / contact_u[0]),
            "linear_relative_residual": result["linear_relative_residual"],
            "weak_displacement_gap_m": result["weak_displacement_gap_m"],
            "force_torque_balance_gap_pa": result["force_torque_balance_gap_pa"],
        }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "results" / "elastic_void_interface_30deg"
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for panels, order in ((128, 4096), (256, 8192), (512, 8192), (1024, 16384)):
        interface = FreeSectorInterface(30.0, panels, 1024, order)
        result = interface.solve(4.5e6)
        metrics = interface.metrics(result)
        case = {
            "contact_panels": panels,
            "maximum_order": order,
            "segment_width_deg": 330.0 / panels,
            "modal_reciprocity_gap": interface.op.maximum_reciprocity_gap,
            "disk_tail_probe_max_relative_gap": interface.op.disk_tail_probe_max_relative_gap,
            **metrics,
        }
        cases.append(case)
        np.savez_compressed(
            output / f"panels{panels}_order{order}.npz",
            edges=interface.edges,
            panel_traction=result["panel_traction"],
            traction_coefficients=result["traction_coefficients"],
            lining_u=result["lining_u"],
            rock_u=result["rock_u"],
        )
        print(json.dumps(case), flush=True)
    reference = cases[-1]
    for case in cases:
        case["mean_pressure_difference_from_finest_percent"] = 100.0 * (
            case["mean_contact_pressure_mpa"] / reference["mean_contact_pressure_mpa"] - 1.0
        )
        case["mean_void_u_difference_from_finest_percent"] = 100.0 * (
            case["mean_void_radial_displacement_mm"] / reference["mean_void_radial_displacement_mm"] - 1.0
        )
    report = {
        "complete": True,
        "scope": "Elastic 4.5 MPa interface verification only; no operating rock plasticity yet.",
        "void_angle_deg": 30.0,
        "cases": cases,
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
