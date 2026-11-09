"""Vectorised classical nonassociated Mohr-Coulomb return mapping."""
from __future__ import annotations

from itertools import combinations
import numpy as np

from mohr_coulomb_nonassociated import yield_halfspaces, _direction_normals


STATE_NAMES = np.array(["elastic", "smooth_face", "edge", "apex"])


def tensors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.zeros((len(values), 3, 3))
    out[:, 0, 0], out[:, 1, 1], out[:, 2, 2] = values[:, 0], values[:, 1], values[:, 2]
    out[:, 0, 1] = out[:, 1, 0] = values[:, 3]
    return out


def components(tensor: np.ndarray) -> np.ndarray:
    return np.column_stack((tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 2, 2], tensor[:, 0, 1]))


class BatchNonassociatedMC:
    def __init__(self, E: float, nu: float, cohesion: float,
                 friction_deg: float, dilation_deg: float):
        if E <= 0.0 or not -1.0 < nu < 0.5:
            raise ValueError("Invalid elastic constants")
        if not 0.0 <= dilation_deg <= friction_deg:
            raise ValueError("Require 0 <= psi <= phi")
        self.E, self.nu, self.c = float(E), float(nu), float(cohesion)
        self.phi, self.psi = float(friction_deg), float(dilation_deg)
        self.normals, self.bound = yield_halfspaces(self.c, self.phi)
        self.flow_normals = _direction_normals(self.psi)
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.C = 2.0 * self.G * np.eye(3) + self.lam * np.ones((3, 3))
        self.sets = {1: [], 2: [], 3: []}
        for count in (1, 2, 3):
            for ids in combinations(range(6), count):
                Ay = self.normals[list(ids)]
                Gp = self.flow_normals[list(ids)]
                matrix = Ay @ self.C @ Gp.T
                if np.linalg.cond(matrix) > 1.0e12:
                    continue
                self.sets[count].append((ids, Ay, Gp, np.linalg.inv(matrix)))

    def update(self, trial: np.ndarray) -> dict:
        trial = np.asarray(trial, dtype=float)
        if trial.ndim != 3 or trial.shape[1:] != (3, 3) or not np.all(np.isfinite(trial)):
            raise ValueError("A finite batch of 3D stress tensors is required")
        if not np.allclose(trial, trial.transpose(0, 2, 1)):
            raise ValueError("Stress tensors must be symmetric")
        values, vectors = np.linalg.eigh(trial)
        count = len(values)
        scale = np.maximum(np.max(np.abs(values), axis=1), max(self.c, 1.0))
        tol = 1.0e-9 * scale
        yield_value = values @ self.normals.T - self.bound
        remaining = np.flatnonzero(np.max(yield_value, axis=1) > tol)
        corrected = values.copy()
        dep = np.zeros_like(values)
        codes = np.zeros(count, dtype=np.int8)

        for active_count in (1, 2, 3):
            if not len(remaining):
                break
            best_measure = np.full(len(remaining), np.inf)
            best_corrected = np.zeros((len(remaining), 3))
            best_dep = np.zeros((len(remaining), 3))
            for unused_ids, Ay, Gp, inverse in self.sets[active_count]:
                multipliers = (values[remaining] @ Ay.T - self.bound) @ inverse.T
                possible = np.min(multipliers, axis=1) >= -1.0e-12
                local_ids = np.flatnonzero(possible)
                if not len(local_ids):
                    continue
                multipliers = np.maximum(multipliers[local_ids], 0.0)
                local_dep = multipliers @ Gp
                local_corrected = values[remaining[local_ids]] - local_dep @ self.C.T
                feasible = np.max(local_corrected @ self.normals.T - self.bound, axis=1) <= tol[remaining[local_ids]]
                feasible &= np.max(np.abs(local_corrected @ Ay.T - self.bound), axis=1) <= tol[remaining[local_ids]]
                local_ids = local_ids[feasible]
                if not len(local_ids):
                    continue
                local_dep = local_dep[feasible]
                local_corrected = local_corrected[feasible]
                measure = np.einsum("ni,ij,nj->n", local_dep, self.C, local_dep)
                improve = measure < best_measure[local_ids]
                selected = local_ids[improve]
                best_measure[selected] = measure[improve]
                best_dep[selected] = local_dep[improve]
                best_corrected[selected] = local_corrected[improve]
            solved_local = np.flatnonzero(np.isfinite(best_measure))
            accepted = remaining[solved_local]
            corrected[accepted] = best_corrected[solved_local]
            dep[accepted] = best_dep[solved_local]
            sorted_stress = np.sort(corrected[accepted], axis=1)
            edge = np.minimum(
                sorted_stress[:, 1] - sorted_stress[:, 0],
                sorted_stress[:, 2] - sorted_stress[:, 1],
            ) < tol[accepted]
            apex = sorted_stress[:, 2] - sorted_stress[:, 0] < tol[accepted]
            codes[accepted] = np.where(apex, 3, np.where(edge, 2, 1))
            solved_mask = np.zeros(count, dtype=bool)
            solved_mask[accepted] = True
            remaining = remaining[~solved_mask[remaining]]
        if len(remaining):
            raise RuntimeError(f"No admissible nonassociated return for {len(remaining)} points")

        stress = (vectors * corrected[:, None, :]) @ vectors.transpose(0, 2, 1)
        plastic = (vectors * dep[:, None, :]) @ vectors.transpose(0, 2, 1)
        elastic = codes == 0
        stress[elastic] = trial[elastic]
        plastic[elastic] = 0.0
        action = (
            2.0 * self.G * plastic
            + self.lam * np.trace(plastic, axis1=1, axis2=2)[:, None, None] * np.eye(3)
        )
        stationarity = np.linalg.norm(trial - stress - action, axis=(1, 2)) / scale
        if float(np.max(stationarity)) > 1.0e-8:
            raise RuntimeError("Batch elastic/plastic balance failed")
        residual = np.max(corrected @ self.normals.T - self.bound, axis=1)
        work = np.sum(stress * plastic, axis=(1, 2))
        return {
            "stress": stress,
            "plastic_strain_increment": plastic,
            "codes": codes,
            "yield_residual_pa": residual,
            "stationarity_relative": stationarity,
            "plastic_work_density": work,
        }
