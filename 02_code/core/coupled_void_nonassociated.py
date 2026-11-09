"""Two-way local-void coupling with nonassociated Mohr--Coulomb rock."""
from __future__ import annotations

import numpy as np

from batch_nonassociated import components, tensors
from rock_wall_eigenstrain import RockWallEigenstrain
from traction_cell_response import sector_average_strain


class CoupledVoidRockDomain:
    """Couple a free-sector lining interface to the full-circle rock domain."""

    def __init__(self, rock, installation: dict, interface, wall_order: int = 1024):
        self.rock, self.interface_operator = rock, interface
        self.installation = {
            key: np.array(installation[key], copy=True)
            for key in ("plastic", "strain", "stress")
        }
        self.installation_wall_displacement_m = float(installation["wall_displacement_m"])
        self.wall = RockWallEigenstrain(
            rock.data["radii"], rock.nt, wall_order, rock.E, rock.nu, rock.R
        )
        self.installation_wall = self.wall.coefficients(self.installation["plastic"])

    def interface(self, internal_pressure_pa: float, plastic: np.ndarray) -> dict:
        free_wall = self.wall.coefficients(plastic) - self.installation_wall
        return self.interface_operator.solve(internal_pressure_pa, free_wall)

    def elastic_traction_strain(self, traction: np.ndarray) -> np.ndarray:
        return sector_average_strain(
            traction, self.rock.data["radii"], self.rock.nt,
            self.rock.E, self.rock.nu, self.rock.R,
        ).reshape(self.rock.n, 4)

    def evaluate(self, internal_pressure_pa: float, plastic: np.ndarray) -> tuple[np.ndarray, dict]:
        boundary = self.interface(internal_pressure_pa, plastic)
        strain = (
            -self.rock.p0 * self.rock.pressure_strain
            + self.rock.source_strain(plastic)
            + self.elastic_traction_strain(boundary["traction_coefficients"])
        )
        return strain, boundary

    def initial_state(self) -> dict:
        strain, boundary = self.evaluate(0.0, self.installation["plastic"])
        gap = float(np.max(np.abs(strain - self.installation["strain"])))
        if gap > 2.0e-12:
            raise RuntimeError(f"Installation state is not reproduced: strain gap={gap:.3e}")
        return {
            "plastic": self.installation["plastic"].copy(),
            "strain": self.installation["strain"].copy(),
            "stress": self.installation["stress"].copy(),
            "internal_pressure_pa": 0.0,
            "boundary": boundary,
            "iterations": 0,
        }

    def step(self, old: dict, internal_pressure_pa: float,
             max_iterations: int = 250, source_tolerance: float = 2.0e-12,
             stress_tolerance_pa: float = 40.0) -> dict:
        trial = old["plastic"].copy()
        memory_x, memory_r, history = [], [], []
        last_boundary = None
        for iteration in range(max_iterations):
            strain, last_boundary = self.evaluate(float(internal_pressure_pa), trial)
            predictor = old["stress"] + (strain - old["strain"]) @ self.rock.C.T
            update = self.rock.material.update(tensors(predictor))
            new = old["plastic"] + components(update["plastic_strain_increment"])
            stress = components(update["stress"])
            gap = float(np.max(np.abs(new - trial)))
            history.append(gap)
            if gap < source_tolerance:
                # Re-evaluate at the accepted plastic state, because interface
                # reactions depend on the current eigenstrain field.
                strain, last_boundary = self.evaluate(float(internal_pressure_pa), new)
                physical = (
                    (strain - new) @ self.rock.C.T
                    + np.tile([-self.rock.p0, -self.rock.p0, -self.rock.p0, 0.0],
                              (self.rock.n, 1))
                )
                mismatch = float(np.max(np.abs(physical - stress)))
                if mismatch > stress_tolerance_pa:
                    raise RuntimeError(f"Coupled constitutive mismatch={mismatch:.6g} Pa")
                return {
                    "plastic": new,
                    "strain": strain,
                    "stress": stress,
                    "internal_pressure_pa": float(internal_pressure_pa),
                    "boundary": last_boundary,
                    "iterations": iteration + 1,
                    "source_gap": gap,
                    "stress_mismatch_pa": mismatch,
                    "state_codes": update["codes"],
                    "yield_residual_pa": float(np.max(update["yield_residual_pa"])),
                    "stationarity_relative": float(np.max(update["stationarity_relative"])),
                    "iteration_history": history,
                }
            residual, vector = (new - trial).ravel(), trial.ravel()
            memory_x.append(vector.copy()); memory_r.append(residual.copy())
            memory_x, memory_r = memory_x[-7:], memory_r[-7:]
            candidate = vector + 0.25 * residual
            if len(memory_x) > 1:
                dx = np.column_stack([b - a for a, b in zip(memory_x[:-1], memory_x[1:])])
                dr = np.column_stack([b - a for a, b in zip(memory_r[:-1], memory_r[1:])])
                gamma = np.linalg.lstsq(dr, residual, rcond=1.0e-10)[0]
                accelerated = candidate - (dx + 0.25 * dr) @ gamma
                if (np.all(np.isfinite(accelerated)) and
                        np.linalg.norm(accelerated - vector) <
                        100.0 * max(np.linalg.norm(residual), 1.0e-12)):
                    candidate = accelerated
                else:
                    memory_x, memory_r = [], []
            trial = candidate.reshape(trial.shape)
        raise RuntimeError(
            f"Coupled void iteration failed at {internal_pressure_pa / 1.0e6:.6g} MPa; "
            f"last gaps={history[-5:]}"
        )

