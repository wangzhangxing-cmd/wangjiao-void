"""Save complete 10-degree void states at each integer pressure from 1 to 10 MPa.

The pressure path is still advanced in 0.1 MPa increments so the nonassociated
Mohr--Coulomb history is identical to the first-yield study.  Integer states are
stored separately for pressure-by-pressure FEM comparisons and plotting.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import traceback

import numpy as np

from coupled_void_nonassociated import CoupledVoidRockDomain
from local_void_interface import FreeSectorInterface
from nonaxisymmetric_nonassociated_domain import FullCircleRockDomain


ANGLE_DEG = 10.0
PRESSURES_MPA = np.arange(0.1, 10.0001, 0.1)


def save_state(path: Path, state: dict) -> None:
    np.savez_compressed(
        path,
        plastic=state["plastic"],
        strain=state["strain"],
        stress=state["stress"],
        panel_traction=state["boundary"]["panel_traction"],
        traction_coefficients=state["boundary"]["traction_coefficients"],
        lining_u=state["boundary"]["lining_u"],
        rock_u=state["boundary"]["rock_u"],
        internal_pressure_pa=np.array(state["internal_pressure_pa"]),
    )


def main() -> None:
    began = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    output = root / "results" / "void_10deg_integer_states"
    output.mkdir(parents=True, exist_ok=True)

    summary_file = output / "summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        expected = [output / f"state_{pressure:02d}MPa.npz" for pressure in range(1, 11)]
        if summary.get("complete") and all(path.exists() for path in expected):
            print(json.dumps({"status": "already_complete", **summary}), flush=True)
            return

    kernel = dict(np.load(
        root / "results" / "elastic_rock_kernel_nr64_nt512" / "kernel.npz"
    ))
    rock = FullCircleRockDomain(kernel)
    saved_installation = np.load(
        root / "results" / "nonaxisymmetric_excavation_nr64_nt512"
        / "installation_state.npz"
    )
    installation = {
        key: saved_installation[key] for key in ("plastic", "strain", "stress")
    }
    installation["wall_displacement_m"] = saved_installation["wall_displacement_m"]
    interface = FreeSectorInterface(ANGLE_DEG, 256, 1024, 8192)
    domain = CoupledVoidRockDomain(rock, installation, interface, 1024)
    state = domain.initial_state()

    progress_file = output / "progress.json"
    restart_file = output / "latest_state.npz"
    records: list[dict] = []
    if progress_file.exists() and restart_file.exists():
        progress = json.loads(progress_file.read_text(encoding="utf-8"))
        records = progress.get("records", [])
        restart = np.load(restart_file)
        state = {key: restart[key] for key in ("plastic", "strain", "stress")}
        state["internal_pressure_pa"] = float(restart["internal_pressure_pa"])
        print(json.dumps({
            "status": "resumed",
            "pressure_mpa": state["internal_pressure_pa"] / 1.0e6,
        }), flush=True)

    last_pressure = records[-1]["pressure_mpa"] if records else 0.0
    try:
        for pressure_mpa in PRESSURES_MPA[PRESSURES_MPA > last_pressure + 1.0e-9]:
            state = domain.step(
                state,
                float(pressure_mpa * 1.0e6),
                max_iterations=800,
                source_tolerance=1.0e-9,
                stress_tolerance_pa=500.0,
            )
            record = {
                "pressure_mpa": float(pressure_mpa),
                "iterations": int(state["iterations"]),
                "source_gap": float(state["source_gap"]),
                "stress_mismatch_pa": float(state["stress_mismatch_pa"]),
                "mean_contact_pressure_mpa": float(
                    np.mean(state["boundary"]["panel_traction"][:, 0]) / 1.0e6
                ),
            }
            records.append(record)
            save_state(restart_file, state)
            if abs(pressure_mpa - round(pressure_mpa)) < 1.0e-9:
                integer_pressure = int(round(pressure_mpa))
                save_state(output / f"state_{integer_pressure:02d}MPa.npz", state)
                print(json.dumps({"saved_integer_state": integer_pressure, **record}), flush=True)
            progress_file.write_text(
                json.dumps({
                    "complete": False,
                    "void_angle_deg": ANGLE_DEG,
                    "pressure_increment_mpa": 0.1,
                    "records": records,
                    "elapsed_seconds": time.perf_counter() - began,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        report = {
            "complete": True,
            "void_angle_deg": ANGLE_DEG,
            "pressure_increment_mpa": 0.1,
            "saved_pressures_mpa": list(range(1, 11)),
            "records": records,
            "elapsed_seconds": time.perf_counter() - began,
        }
        summary_file.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("COMPLETE", json.dumps(report, ensure_ascii=False), flush=True)
    except Exception as exc:
        failure = {
            "complete": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "records": records,
        }
        (output / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
