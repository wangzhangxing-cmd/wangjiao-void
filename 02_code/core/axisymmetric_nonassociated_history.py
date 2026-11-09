"""Axisymmetric excavation-installation-operation history for the new model.

The rock is an infinite plane-strain medium.  Plastic eigenstrains are piecewise
constant in logarithmically spaced radial annuli; the exterior beyond the
plastic-source domain remains an exact elastic tail.  Stress is tensile
positive.  Wall pressure q and internal gas pressure pi are positive.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
import numpy as np

from layered_ring_operator import LiningInput, SealLiningOperator
from mohr_coulomb_nonassociated import return_map, yield_halfspaces


@dataclass(frozen=True)
class RockInput:
    elastic_modulus_pa: float = 12.0e9
    poisson_ratio: float = 0.26
    cohesion_pa: float = 0.9e6
    friction_angle_deg: float = 40.0
    dilation_angle_deg: float = 10.0
    initial_stress_pa: float = 3.75e6
    cavern_radius_m: float = 6.5


class RadialRockDomain:
    def __init__(self, rock: RockInput = RockInput(), radial_cells: int = 64,
                 source_extent_ratio: float = 2.0):
        self.rock = rock
        self.nr = int(radial_cells)
        self.extent = float(source_extent_ratio)
        if self.nr < 4 or self.extent <= 1.0:
            raise ValueError("Require at least four cells and source extent > 1")
        E, nu = rock.elastic_modulus_pa, rock.poisson_ratio
        self.G = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.C = 2.0 * self.G * np.eye(3) + self.lam * np.ones((3, 3))
        self.radii = np.geomspace(1.0, self.extent, self.nr + 1)
        self.centres = np.sqrt(self.radii[:-1] * self.radii[1:])
        P = 2.0 * (self.lam + self.G)
        Q = 2.0 * self.G
        M = self.lam + 2.0 * self.G

        H = np.zeros((2 * self.nr, 2 * self.nr))
        J = np.zeros((2 * self.nr, 1 + 3 * self.nr))
        source = np.zeros((self.nr, 3, 1 + 3 * self.nr))
        for i in range(self.nr):
            source[i, :, 1 + 3 * i:1 + 3 * (i + 1)] = self.C
        particular = (source[:, 0] - source[:, 1]) / (2.0 * M)
        H[0, :2] = [P / E, -Q / E]
        J[0, 0] = -1.0 / E
        J[0] += source[0, 0] / E - particular[0] * M / E
        row = 1
        for i in range(self.nr - 1):
            x = self.radii[i + 1]
            logx = np.log(x)
            H[row, 2 * i:2 * i + 2] = [x, 1.0 / x]
            H[row, 2 * i + 2:2 * i + 4] = [-x, -1.0 / x]
            J[row] = -(particular[i] - particular[i + 1]) * x * logx
            row += 1
            H[row, 2 * i:2 * i + 2] = [P / E, -Q / (E * x * x)]
            H[row, 2 * i + 2:2 * i + 4] = [-P / E, Q / (E * x * x)]
            J[row] = (
                source[i, 0] - source[i + 1, 0]
                - (particular[i] - particular[i + 1]) * (P * logx + M)
            ) / E
            row += 1
        x = self.radii[-1]
        H[row, -2] = (P + Q) / E
        J[row] = (
            source[-1, 0]
            - particular[-1] * ((P + Q) * np.log(x) + M)
        ) / E
        self.coefficients = np.linalg.solve(H, J).reshape(self.nr, 2, -1)
        self.strain_map = np.zeros((self.nr, 3, 1 + 3 * self.nr))
        for i, x in enumerate(self.centres):
            A, B = self.coefficients[i]
            p = particular[i]
            logx = np.log(x)
            self.strain_map[i, 0] = A - B / x ** 2 + p * (logx + 1.0)
            self.strain_map[i, 1] = A + B / x ** 2 + p * logx
        self.wall_map = rock.cavern_radius_m * (
            self.coefficients[0, 0] + self.coefficients[0, 1]
        )

    def initial_state(self) -> dict:
        return {
            "plastic": np.zeros((self.nr, 3)),
            "strain": np.zeros((self.nr, 3)),
            "stress": np.full((self.nr, 3), -self.rock.initial_stress_pa),
            "wall_displacement_m": 0.0,
            "wall_pressure_pa": self.rock.initial_stress_pa,
            "iterations": 0,
            "source_gap": 0.0,
            "stress_mismatch_pa": 0.0,
            "states": ["elastic"] * self.nr,
        }

    def evaluate(self, wall_pressure_pa: float, plastic: np.ndarray) -> tuple[np.ndarray, float]:
        increment = wall_pressure_pa - self.rock.initial_stress_pa
        vector = np.concatenate(([increment], np.asarray(plastic).ravel()))
        strain = np.einsum("ijk,k->ij", self.strain_map, vector)
        displacement = float(self.wall_map @ vector)
        return strain, displacement

    def step(self, old: dict, wall_pressure_pa: float | None = None,
             gas_pressure_pa: float | None = None, installation: dict | None = None,
             lining_relation: tuple[float, float] | None = None,
             max_iterations: int = 400) -> dict:
        coupled = gas_pressure_pa is not None
        if coupled != (installation is not None and lining_relation is not None):
            raise ValueError("Coupled operation requires gas pressure, installation and lining relation")
        if not coupled and wall_pressure_pa is None:
            raise ValueError("Prescribed wall pressure is required")
        trial = old["plastic"].copy()
        memory_x, memory_r, history = [], [], []
        q = float(old["wall_pressure_pa"] if coupled else wall_pressure_pa)
        for iteration in range(max_iterations):
            if coupled:
                alpha, beta = lining_relation
                source_u = float(self.wall_map[1:] @ (trial - installation["plastic"]).ravel())
                q = (alpha * float(gas_pressure_pa) - source_u) / (beta + self.wall_map[0])
            strain, displacement = self.evaluate(q, trial)
            new = np.zeros_like(trial)
            stress = np.zeros_like(trial)
            states = []
            for i in range(self.nr):
                increment = strain[i] - old["strain"][i]
                predictor = old["stress"][i] + self.C @ increment
                response = return_map(
                    np.diag(predictor),
                    self.rock.elastic_modulus_pa,
                    self.rock.poisson_ratio,
                    self.rock.cohesion_pa,
                    self.rock.friction_angle_deg,
                    self.rock.dilation_angle_deg,
                )
                new[i] = old["plastic"][i] + np.diag(response["plastic_strain_increment"])
                stress[i] = np.diag(response["stress"])
                states.append(response["state"])
            gap = float(np.max(np.abs(new - trial)))
            history.append(gap)
            if gap < 2.0e-12:
                physical = -self.rock.initial_stress_pa + np.einsum(
                    "ij,nj->ni", self.C, strain - trial
                )
                mismatch = float(np.max(np.abs(physical - stress)))
                if mismatch > 20.0:
                    raise RuntimeError("Converged constitutive mismatch is too large")
                return {
                    "plastic": new,
                    "strain": strain,
                    "stress": stress,
                    "wall_displacement_m": displacement,
                    "wall_pressure_pa": q,
                    "gas_pressure_pa": gas_pressure_pa,
                    "iterations": iteration + 1,
                    "source_gap": gap,
                    "stress_mismatch_pa": mismatch,
                    "states": states,
                    "iteration_history": history,
                }
            residual = (new - trial).ravel()
            vector = trial.ravel()
            memory_x.append(vector.copy())
            memory_r.append(residual.copy())
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
            f"Nonassociated radial iteration failed at q={q / 1e6:.6g} MPa; "
            f"last gaps={history[-5:]}"
        )

    def state_checks(self, state: dict) -> dict:
        normals, bound = yield_halfspaces(
            self.rock.cohesion_pa, self.rock.friction_angle_deg
        )
        yield_residual = max(
            float(np.max(normals @ np.sort(s) - bound)) for s in state["stress"]
        )
        plastic_norm = np.linalg.norm(state["plastic"], axis=1)
        active = np.flatnonzero(plastic_norm > 1.0e-10)
        return {
            "yield_residual_pa": yield_residual,
            "plastic_cell_count": int(len(active)),
            "plastic_outer_radius_m": (
                float(self.rock.cavern_radius_m * self.radii[active[-1] + 1])
                if len(active) else self.rock.cavern_radius_m
            ),
            "outermost_cell_plastic_norm": float(plastic_norm[-1]),
            "maximum_plastic_norm": float(np.max(plastic_norm)),
        }


def lining_axisymmetric_relation() -> tuple[float, float, dict]:
    operator = SealLiningOperator(LiningInput(
        residual_hoop_ratio=0.05, residual_shear_ratio=0.05
    ))
    mode = operator.mode(0)
    K = float(mode["K"][0, 0].real)
    L = float(mode["L"][0, 0].real)
    alpha = -L / K
    beta = 1.0 / K
    return alpha, beta, {
        "K_pa_per_m": K,
        "L_dimensionless": L,
        "alpha_m_per_pa": alpha,
        "beta_m_per_pa": beta,
    }


def run_case(radial_cells: int, extent: float, excavation_steps: int = 30,
             pressure_increment_mpa: float = 0.25, target_pressure_mpa: float = 10.0,
             state_output: Path | None = None) -> dict:
    began = time.perf_counter()
    domain = RadialRockDomain(radial_cells=radial_cells, source_extent_ratio=extent)
    alpha, beta, relation = lining_axisymmetric_relation()
    elastic_u = domain.evaluate(domain.rock.initial_stress_pa + 1.0e6,
                                np.zeros((radial_cells, 3)))[1]
    elastic_exact = domain.rock.cavern_radius_m * 1.0e6 / (2.0 * domain.G)
    if abs(elastic_u / elastic_exact - 1.0) > 1.0e-10:
        raise AssertionError("Elastic wall compliance check failed")

    state = domain.initial_state()
    excavation_records = []
    for index, q in enumerate(np.linspace(domain.rock.initial_stress_pa, 0.0, excavation_steps + 1)[1:], 1):
        state = domain.step(state, wall_pressure_pa=float(q))
        excavation_records.append({
            "step": index,
            "wall_pressure_mpa": q / 1.0e6,
            "iterations": state["iterations"],
            "wall_displacement_mm": state["wall_displacement_m"] * 1000.0,
        })
    installation = state
    installation_checks = domain.state_checks(installation)

    operation_records = []
    first_operating_plasticity = None
    pressures = np.arange(pressure_increment_mpa, target_pressure_mpa + 0.5 * pressure_increment_mpa,
                          pressure_increment_mpa)
    for index, pi_mpa in enumerate(pressures, 1):
        previous = state
        state = domain.step(
            state,
            gas_pressure_pa=float(pi_mpa * 1.0e6),
            installation=installation,
            lining_relation=(alpha, beta),
        )
        checks = domain.state_checks(state)
        plastic_increment = state["plastic"] - previous["plastic"]
        increment_norm = np.linalg.norm(plastic_increment, axis=1)
        changed_from_install = np.linalg.norm(
            state["plastic"] - installation["plastic"], axis=1
        )
        incremental_cells = int(np.sum(increment_norm > 1.0e-12))
        if incremental_cells and first_operating_plasticity is None:
            first_operating_plasticity = {
                "previous_gas_pressure_mpa": float(pi_mpa - pressure_increment_mpa),
                "first_accepted_gas_pressure_mpa": float(pi_mpa),
                "cells": incremental_cells,
                "maximum_increment_norm": float(np.max(increment_norm)),
            }
        operation_u = state["wall_displacement_m"] - installation["wall_displacement_m"]
        lining_u = alpha * pi_mpa * 1.0e6 - beta * state["wall_pressure_pa"]
        operation_records.append({
            "step": index,
            "gas_pressure_mpa": float(pi_mpa),
            "interface_pressure_mpa": state["wall_pressure_pa"] / 1.0e6,
            "wall_displacement_increment_mm": operation_u * 1000.0,
            "lining_displacement_mm": lining_u * 1000.0,
            "interface_displacement_gap_mm": (operation_u - lining_u) * 1000.0,
            "iterations": state["iterations"],
            "source_gap": state["source_gap"],
            "stress_mismatch_pa": state["stress_mismatch_pa"],
            "plastic_increment_cell_count": incremental_cells,
            "maximum_plastic_increment_norm": float(np.max(increment_norm)),
            "plastic_change_from_install_cell_count": int(np.sum(changed_from_install > 1.0e-12)),
            "maximum_plastic_change_from_install_norm": float(np.max(changed_from_install)),
            **checks,
        })
    if state_output is not None:
        state_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            state_output,
            radii_ratio=domain.radii,
            centres_ratio=domain.centres,
            installation_plastic=installation["plastic"],
            installation_strain=installation["strain"],
            installation_stress=installation["stress"],
            installation_wall_displacement_m=np.array(installation["wall_displacement_m"]),
            final_plastic=state["plastic"],
            final_strain=state["strain"],
            final_stress=state["stress"],
            final_wall_displacement_m=np.array(state["wall_displacement_m"]),
            final_wall_pressure_pa=np.array(state["wall_pressure_pa"]),
        )
    return {
        "complete": True,
        "radial_cells": radial_cells,
        "source_extent_ratio": extent,
        "excavation_steps": excavation_steps,
        "pressure_increment_mpa": pressure_increment_mpa,
        "target_pressure_mpa": target_pressure_mpa,
        "rock_input": domain.rock.__dict__,
        "lining_relation": relation,
        "elastic_compliance_relative_error": abs(elastic_u / elastic_exact - 1.0),
        "installation": {
            "wall_displacement_mm": installation["wall_displacement_m"] * 1000.0,
            **installation_checks,
        },
        "first_operating_plasticity": first_operating_plasticity,
        "excavation_records": excavation_records,
        "operation_records": operation_records,
        "elapsed_seconds": time.perf_counter() - began,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "results" / "axisymmetric_nonassociated_baseline"
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for nr, extent in ((32, 2.0), (64, 2.0), (64, 3.0)):
        result = run_case(
            nr,
            extent,
            state_output=output / f"nr{nr}_extent{extent:.0f}_states.npz",
        )
        runs.append(result)
        (output / f"nr{nr}_extent{extent:.0f}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        final = result["operation_records"][-1]
        print(json.dumps({
            "radial_cells": nr,
            "extent": extent,
            "final_interface_pressure_mpa": final["interface_pressure_mpa"],
            "final_wall_u_mm": final["wall_displacement_increment_mm"],
            "plastic_outer_radius_m": final["plastic_outer_radius_m"],
        }), flush=True)
    reference = runs[1]["operation_records"][-1]
    comparisons = []
    for run in runs:
        final = run["operation_records"][-1]
        comparisons.append({
            "radial_cells": run["radial_cells"],
            "extent": run["source_extent_ratio"],
            "interface_pressure_difference_percent": 100.0 * (
                final["interface_pressure_mpa"] / reference["interface_pressure_mpa"] - 1.0
            ),
            "wall_displacement_difference_percent": 100.0 * (
                final["wall_displacement_increment_mm"] / reference["wall_displacement_increment_mm"] - 1.0
            ),
            "plastic_outer_radius_difference_percent": 100.0 * (
                final["plastic_outer_radius_m"] / reference["plastic_outer_radius_m"] - 1.0
            ),
        })
    summary = {"runs": runs, "final_10mpa_comparison": comparisons}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
