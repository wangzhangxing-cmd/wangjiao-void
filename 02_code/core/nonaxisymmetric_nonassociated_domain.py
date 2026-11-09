"""Full-circle nonaxisymmetric rock domain with nonassociated plasticity."""
from __future__ import annotations

import numpy as np

from batch_nonassociated import BatchNonassociatedMC, components, tensors


def tensor_basis() -> tuple[np.ndarray, ...]:
    xx = np.diag([1.0, 0.0, 0.0])
    yy = np.diag([0.0, 1.0, 0.0])
    zz = np.diag([0.0, 0.0, 1.0])
    xy = np.zeros((3, 3)); xy[0, 1] = xy[1, 0] = 1.0
    return xx, yy, zz, xy


def rotation(angle: float) -> np.ndarray:
    q = np.eye(3)
    q[:2, :2] = [[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]]
    return q


def transform(q: np.ndarray) -> np.ndarray:
    return np.column_stack([components((q @ b @ q.T)[None])[0] for b in tensor_basis()])


class FullCircleRockDomain:
    def __init__(self, kernel: dict, E: float = 12.0e9, nu: float = 0.26,
                 cohesion: float = 0.9e6, friction_deg: float = 40.0,
                 dilation_deg: float = 10.0, initial_stress: float = 3.75e6,
                 cavern_radius: float = 6.5):
        self.data = kernel
        self.nr, self.nt = int(kernel["nr"]), int(kernel["nt"])
        self.n = self.nr * self.nt
        self.E, self.nu, self.p0, self.R = E, nu, initial_stress, cavern_radius
        self.G = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.C = np.array([
            [lam + 2.0 * self.G, lam, lam, 0.0],
            [lam, lam + 2.0 * self.G, lam, 0.0],
            [lam, lam, lam + 2.0 * self.G, 0.0],
            [0.0, 0.0, 0.0, 2.0 * self.G],
        ])
        self.angles = np.asarray(kernel["angles"])
        self.into = np.array([transform(rotation(a).T) for a in self.angles])
        self.out = np.array([transform(rotation(a)) for a in self.angles])
        self.frequency_kernel = np.fft.rfft(kernel["K"], axis=2)
        self.wall_map = np.asarray(kernel["wall_map"])
        radii = np.asarray(kernel["radii"])
        width = 2.0 * np.pi / self.nt
        value = (
            self.R ** 2 / (2.0 * self.G)
            * 2.0 * np.log(radii[1:] / radii[:-1])
            / (radii[1:] ** 2 - radii[:-1] ** 2)
            * np.sin(width) / width
        )
        local = np.zeros((self.nr, self.nt, 4))
        local[:, :, 0] = -value[:, None]
        local[:, :, 1] = value[:, None]
        self.pressure_strain = np.einsum("jab,rjb->rja", self.out, local).reshape(self.n, 4)
        self.material = BatchNonassociatedMC(E, nu, cohesion, friction_deg, dilation_deg)

    def initial_state(self) -> dict:
        return {
            "plastic": np.zeros((self.n, 4)),
            "strain": np.zeros((self.n, 4)),
            "stress": np.tile([-self.p0, -self.p0, -self.p0, 0.0], (self.n, 1)),
            "wall_displacement_m": 0.0,
            "wall_pressure_pa": self.p0,
            "iterations": 0,
        }

    def source_strain(self, plastic: np.ndarray) -> np.ndarray:
        local = np.einsum("jab,rjb->rja", self.into, plastic.reshape(self.nr, self.nt, 4))
        frequency = np.fft.rfft(local, axis=1)
        response = np.einsum(
            "rskab,skb->rka", self.frequency_kernel, frequency, optimize=True
        )
        local_response = np.fft.irfft(response, n=self.nt, axis=1)
        return np.einsum("jab,rjb->rja", self.out, local_response).reshape(self.n, 4)

    def evaluate(self, wall_pressure_pa: float, plastic: np.ndarray) -> tuple[np.ndarray, float]:
        strain = self.pressure_strain * (wall_pressure_pa - self.p0) + self.source_strain(plastic)
        displacement = (
            (wall_pressure_pa - self.p0) * self.R / (2.0 * self.G)
            + self.wall_map @ plastic.ravel()
        )
        return strain, float(displacement)

    def step(self, old: dict, wall_pressure_pa: float, max_iterations: int = 350) -> dict:
        trial = old["plastic"].copy()
        memory_x, memory_r, history = [], [], []
        for iteration in range(max_iterations):
            strain, displacement = self.evaluate(wall_pressure_pa, trial)
            predictor = old["stress"] + (strain - old["strain"]) @ self.C.T
            result = self.material.update(tensors(predictor))
            new = old["plastic"] + components(result["plastic_strain_increment"])
            stress = components(result["stress"])
            gap = float(np.max(np.abs(new - trial)))
            history.append(gap)
            if gap < 2.0e-12:
                physical = (
                    (strain - trial) @ self.C.T
                    + np.tile([-self.p0, -self.p0, -self.p0, 0.0], (self.n, 1))
                )
                mismatch = float(np.max(np.abs(physical - stress)))
                if mismatch > 20.0:
                    raise RuntimeError("Full-circle constitutive mismatch is too large")
                return {
                    "plastic": new,
                    "strain": strain,
                    "stress": stress,
                    "wall_displacement_m": displacement,
                    "wall_pressure_pa": float(wall_pressure_pa),
                    "iterations": iteration + 1,
                    "source_gap": gap,
                    "stress_mismatch_pa": mismatch,
                    "state_codes": result["codes"],
                    "yield_residual_pa": float(np.max(result["yield_residual_pa"])),
                    "stationarity_relative": float(np.max(result["stationarity_relative"])),
                    "iteration_history": history,
                }
            residual = (new - trial).ravel()
            vector = trial.ravel()
            memory_x.append(vector.copy()); memory_r.append(residual.copy())
            memory_x, memory_r = memory_x[-7:], memory_r[-7:]
            candidate = vector + 0.35 * residual
            if len(memory_x) > 1:
                dx = np.column_stack([b - a for a, b in zip(memory_x[:-1], memory_x[1:])])
                dr = np.column_stack([b - a for a, b in zip(memory_r[:-1], memory_r[1:])])
                gamma = np.linalg.lstsq(dr, residual, rcond=1.0e-10)[0]
                accelerated = candidate - (dx + 0.35 * dr) @ gamma
                if (np.all(np.isfinite(accelerated)) and
                        np.linalg.norm(accelerated - vector) < 100.0 * max(np.linalg.norm(residual), 1.0e-12)):
                    candidate = accelerated
                else:
                    memory_x, memory_r = [], []
            trial = candidate.reshape(trial.shape)
        raise RuntimeError(
            f"Full-circle iteration failed at q={wall_pressure_pa / 1e6:.6g} MPa; "
            f"last gaps={history[-5:]}"
        )

    def local_components(self, values: np.ndarray) -> np.ndarray:
        return np.einsum(
            "jab,rjb->rja", self.into, values.reshape(self.nr, self.nt, 4)
        )
