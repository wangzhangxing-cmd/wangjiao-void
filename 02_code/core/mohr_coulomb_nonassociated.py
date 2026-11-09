"""Classical 3D nonassociated Mohr-Coulomb material-point update.

Sign convention: tensile stress is positive.  The yield surface uses friction
angle phi and the plastic potential uses dilation angle psi.  Both are written
as six principal-stress half-spaces.  A small active-set solve treats smooth
faces, edges and the apex without reducing the constitutive law to 2D.

This is the classical nonsmooth model used by the semi-analytical solver.  It
does not claim to reproduce Abaqus' smoothed deviatoric/hyperbolic potential
exactly; that difference must be quantified in the validation chapter.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np


def elastic_action(strain: np.ndarray, E: float, nu: float) -> np.ndarray:
    G = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return 2.0 * G * strain + lam * np.trace(strain) * np.eye(3)


def _direction_normals(angle_deg: float) -> np.ndarray:
    if not 0.0 <= angle_deg < 89.0:
        raise ValueError("angle must satisfy 0 <= angle < 89 degrees")
    s = np.sin(np.deg2rad(angle_deg))
    normals = []
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            v = np.zeros(3)
            v[i] = 1.0 + s
            v[j] = -(1.0 - s)
            normals.append(v)
    return np.asarray(normals)


def yield_halfspaces(cohesion: float, friction_deg: float) -> tuple[np.ndarray, float]:
    if cohesion < 0.0 or not 0.0 < friction_deg < 89.0:
        raise ValueError("Require c>=0 and 0<phi<89 degrees")
    return _direction_normals(friction_deg), 2.0 * cohesion * np.cos(np.deg2rad(friction_deg))


def return_map(
    trial: np.ndarray,
    E: float,
    nu: float,
    cohesion: float,
    friction_deg: float,
    dilation_deg: float,
) -> dict:
    if E <= 0.0 or not -1.0 < nu < 0.5:
        raise ValueError("Invalid elastic constants")
    if not 0.0 <= dilation_deg <= friction_deg:
        raise ValueError("Current model requires 0 <= psi <= phi")
    trial = np.asarray(trial, dtype=float)
    if trial.shape != (3, 3) or not np.allclose(trial, trial.T):
        raise ValueError("A symmetric 3D trial stress is required")

    values, vectors = np.linalg.eigh(trial)
    Aall, bound = yield_halfspaces(cohesion, friction_deg)
    Gall = _direction_normals(dilation_deg)
    scale = max(float(np.max(np.abs(values))), cohesion, 1.0)
    tol = 1.0e-9 * scale
    trial_yield = Aall @ values - bound
    if float(np.max(trial_yield)) <= tol:
        return {
            "stress": trial.copy(),
            "plastic_strain_increment": np.zeros((3, 3)),
            "state": "elastic",
            "active_indices": [],
            "multipliers": np.zeros(0),
            "yield_residual_pa": float(np.max(trial_yield)),
            "stationarity_relative": 0.0,
            "plastic_work_density": 0.0,
        }

    Gmod = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C = 2.0 * Gmod * np.eye(3) + lam * np.ones((3, 3))
    candidates = []
    for active_count in (1, 2, 3):
        for ids in combinations(range(6), active_count):
            Ay = Aall[list(ids)]
            Gp = Gall[list(ids)]
            M = Ay @ C @ Gp.T
            if np.linalg.cond(M) > 1.0e12:
                continue
            multipliers = np.linalg.solve(M, Ay @ values - bound)
            if float(np.min(multipliers)) < -1.0e-12:
                continue
            multipliers = np.maximum(multipliers, 0.0)
            dep = Gp.T @ multipliers
            corrected = values - C @ dep
            residual = Aall @ corrected - bound
            if float(np.max(residual)) > tol:
                continue
            if float(np.max(np.abs(Ay @ corrected - bound))) > tol:
                continue
            # Nonassociated flow has no closest-point minimisation principle.
            # Rank the admissible KKT states by the elastic correction norm;
            # a unique minimum is required below.
            correction_measure = float(dep @ C @ dep)
            candidates.append((correction_measure, ids, multipliers, corrected, dep))
        if candidates:
            break
    if not candidates:
        raise RuntimeError("No admissible nonassociated Mohr-Coulomb return found")
    candidates.sort(key=lambda item: item[0])
    if len(candidates) > 1:
        first, second = candidates[0][0], candidates[1][0]
        if abs(second - first) <= 1.0e-12 * max(abs(first), 1.0):
            if not np.allclose(candidates[0][3], candidates[1][3], rtol=1e-10, atol=tol):
                raise RuntimeError("Ambiguous nonassociated active set")
    unused, ids, multipliers, corrected, dep = candidates[0]

    stress = (vectors * corrected) @ vectors.T
    plastic = (vectors * dep) @ vectors.T
    eigenvalues = np.linalg.eigvalsh(stress)
    if eigenvalues[-1] - eigenvalues[0] < tol:
        state = "apex"
    elif min(eigenvalues[1] - eigenvalues[0], eigenvalues[2] - eigenvalues[1]) < tol:
        state = "edge"
    else:
        state = "smooth_face"
    stationarity = np.linalg.norm(trial - stress - elastic_action(plastic, E, nu)) / scale
    if stationarity > 1.0e-8:
        raise RuntimeError("Elastic/plastic stress balance failed")
    work = float(np.sum(stress * plastic))
    return {
        "stress": stress,
        "plastic_strain_increment": plastic,
        "state": state,
        "active_indices": list(ids),
        "multipliers": multipliers,
        "yield_residual_pa": float(np.max(Aall @ eigenvalues - bound)),
        "stationarity_relative": float(stationarity),
        "plastic_work_density": work,
    }


def update(
    old_stress: np.ndarray,
    strain_increment: np.ndarray,
    E: float,
    nu: float,
    cohesion: float,
    friction_deg: float,
    dilation_deg: float,
) -> dict:
    trial = old_stress + elastic_action(strain_increment, E, nu)
    return return_map(trial, E, nu, cohesion, friction_deg, dilation_deg)
