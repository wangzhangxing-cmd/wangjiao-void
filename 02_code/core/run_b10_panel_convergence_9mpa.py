"""Interface-panel convergence study for the 10-degree void at 9 MPa.

All cases use the same 64x512 rock domain, 0.1 MPa pressure history, modal
orders and material parameters.  Only the number of piecewise-constant contact
panels changes.  The established 256-panel integer-pressure state is reused.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
import traceback

import numpy as np

from coupled_void_nonassociated import CoupledVoidRockDomain
from local_void_interface import FreeSectorInterface, arc_mean
from nonaxisymmetric_nonassociated_domain import FullCircleRockDomain
from seal_stress_recovery import recover_concrete_face_stress, recover_seal_stress


ANGLE_DEG = 10.0
TARGET_PRESSURE_MPA = 9.0
PANEL_COUNTS = (128, 256, 512, 1024)
PRESSURES_MPA = np.arange(0.1, TARGET_PRESSURE_MPA + 0.0001, 0.1)
CORE_ORDER = 1024
MAXIMUM_ORDER = 8192
WALL_ORDER = 1024


def save_state(path: Path, state: dict) -> None:
    np.savez_compressed(
        path,
        plastic=state["plastic"], strain=state["strain"], stress=state["stress"],
        panel_traction=state["boundary"]["panel_traction"],
        traction_coefficients=state["boundary"]["traction_coefficients"],
        lining_u=state["boundary"]["lining_u"], rock_u=state["boundary"]["rock_u"],
        internal_pressure_pa=np.array(state["internal_pressure_pa"]),
    )


def load_core_state(path: Path) -> dict:
    saved = np.load(path)
    return {
        "plastic": saved["plastic"], "strain": saved["strain"],
        "stress": saved["stress"],
        "internal_pressure_pa": float(saved["internal_pressure_pa"]),
    }


def load_full_state(path: Path) -> dict:
    saved = np.load(path)
    return {
        "plastic": saved["plastic"], "strain": saved["strain"],
        "stress": saved["stress"],
        "internal_pressure_pa": float(saved["internal_pressure_pa"]),
        "boundary": {
            "panel_traction": saved["panel_traction"],
            "traction_coefficients": saved["traction_coefficients"],
            "lining_u": saved["lining_u"], "rock_u": saved["rock_u"],
        },
    }


def final_metrics(domain: CoupledVoidRockDomain, interface: FreeSectorInterface,
                  state: dict, installation: dict, reused: bool,
                  records: list[dict], elapsed_seconds: float) -> dict:
    boundary = state["boundary"]
    theta = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    signed = (np.rad2deg(theta) + 180.0) % 360.0 - 180.0
    endpoint_distance = np.minimum(
        np.abs(signed - ANGLE_DEG / 2.0), np.abs(signed + ANGLE_DEG / 2.0)
    )
    seal = recover_seal_stress(
        interface, boundary["lining_u"], TARGET_PRESSURE_MPA * 1.0e6, theta
    )
    seal_mises = np.maximum(seal["inner"]["mises"], seal["outer"]["mises"]) / 1.0e6
    concrete = recover_concrete_face_stress(
        interface, boundary["lining_u"], TARGET_PRESSURE_MPA * 1.0e6, theta
    )
    concrete_max = np.max(
        np.column_stack([item["maximum_principal"] for item in concrete.values()]), axis=1
    ) / 1.0e6

    increment = state["plastic"] - installation["plastic"]
    local = domain.rock.local_components(increment)
    proxy = np.sqrt(np.sum(local[..., :3] ** 2, axis=-1) + 2.0 * local[..., 3] ** 2)
    radii = np.asarray(domain.rock.data["radii"])
    area = np.broadcast_to(
        ((radii[1:] ** 2 - radii[:-1] ** 2) * 0.5)[:, None]
        * (2.0 * np.pi / domain.rock.nt), proxy.shape,
    )
    active = proxy > 1.0e-6
    outer_radius = float(np.max(radii[1:][np.any(active, axis=1)])) if np.any(active) else 0.0
    weak_gap = float(boundary.get("weak_displacement_gap_m", np.nan))
    force_gap = float(boundary.get("force_torque_balance_gap_pa", np.nan))
    return {
        "complete": True,
        "void_angle_deg": ANGLE_DEG,
        "pressure_mpa": TARGET_PRESSURE_MPA,
        "contact_panels": interface.panels,
        "core_order": CORE_ORDER,
        "maximum_order": MAXIMUM_ORDER,
        "rock_nr": len(radii) - 1,
        "rock_nt": domain.rock.nt,
        "reused_existing_state": reused,
        "mean_contact_pressure_mpa": float(
            np.mean(boundary["panel_traction"][:, 0]) / 1.0e6
        ),
        "mean_contact_ur_mm": float(
            arc_mean(boundary["lining_u"], interface.alpha, 2.0 * np.pi - interface.alpha)[0]
            * 1000.0
        ),
        "mean_void_ur_mm": float(
            arc_mean(boundary["lining_u"], -interface.alpha, interface.alpha)[0] * 1000.0
        ),
        "seal_mises_max_mpa": float(np.max(seal_mises)),
        "seal_mises_p99_mpa": float(np.quantile(seal_mises, 0.99)),
        "seal_mises_max_exclude5deg_mpa": float(np.max(seal_mises[endpoint_distance > 5.0])),
        "seal_mises_max_exclude10deg_mpa": float(np.max(seal_mises[endpoint_distance > 10.0])),
        "concrete_max_principal_max_mpa": float(np.max(concrete_max)),
        "concrete_max_principal_p99_mpa": float(np.quantile(concrete_max, 0.99)),
        "rock_plastic_area_gt_1e6_m2": float(np.sum(area[active])),
        "rock_plastic_proxy_max": float(np.max(proxy)),
        "rock_plastic_outer_radius_m": outer_radius,
        "linear_relative_residual": float(boundary.get("linear_relative_residual", np.nan)),
        "weak_displacement_gap_m": weak_gap,
        "force_torque_balance_gap_pa": force_gap,
        "max_source_gap": float(max((row["source_gap"] for row in records), default=np.nan)),
        "max_stress_mismatch_pa": float(
            max((row["stress_mismatch_pa"] for row in records), default=np.nan)
        ),
        "elapsed_seconds": elapsed_seconds,
    }


def run_case(root: Path, rock: FullCircleRockDomain, installation: dict,
             panels: int, output: Path) -> dict:
    began = time.perf_counter()
    case = output / f"panels_{panels}"
    case.mkdir(parents=True, exist_ok=True)
    summary_file = case / "summary.json"
    if summary_file.exists():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary.get("complete"):
            print(json.dumps({"panels": panels, "status": "already_complete"}), flush=True)
            return summary

    interface = FreeSectorInterface(
        ANGLE_DEG, panels, CORE_ORDER, MAXIMUM_ORDER
    )
    domain = CoupledVoidRockDomain(rock, installation, interface, WALL_ORDER)
    records: list[dict] = []
    reused = panels == 256
    if reused:
        state = load_full_state(
            root / "results" / "void_10deg_integer_states" / "state_09MPa.npz"
        )
        # Re-evaluate the interface to attach solver residual diagnostics.
        boundary = domain.interface(state["internal_pressure_pa"], state["plastic"])
        state["boundary"] = boundary
    else:
        state = domain.initial_state()
        progress_file = case / "progress.json"
        restart_file = case / "latest_state.npz"
        if progress_file.exists() and restart_file.exists():
            progress = json.loads(progress_file.read_text(encoding="utf-8"))
            records = progress.get("records", [])
            state = load_core_state(restart_file)
            print(json.dumps({
                "panels": panels, "status": "resumed",
                "pressure_mpa": state["internal_pressure_pa"] / 1.0e6,
            }), flush=True)
        last = records[-1]["pressure_mpa"] if records else 0.0
        for pressure_mpa in PRESSURES_MPA[PRESSURES_MPA > last + 1.0e-9]:
            state = domain.step(
                state, float(pressure_mpa * 1.0e6), max_iterations=800,
                source_tolerance=1.0e-9, stress_tolerance_pa=500.0,
            )
            record = {
                "pressure_mpa": float(pressure_mpa),
                "iterations": int(state["iterations"]),
                "source_gap": float(state["source_gap"]),
                "stress_mismatch_pa": float(state["stress_mismatch_pa"]),
            }
            records.append(record)
            save_state(restart_file, state)
            progress_file.write_text(json.dumps({
                "complete": False, "contact_panels": panels, "records": records,
                "elapsed_seconds": time.perf_counter() - began,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            if abs(pressure_mpa - round(pressure_mpa)) < 1.0e-9:
                print(json.dumps({"panels": panels, **record}), flush=True)
        save_state(case / "state_09MPa.npz", state)

    report = final_metrics(
        domain, interface, state, installation, reused, records,
        time.perf_counter() - began,
    )
    summary_file.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"panels": panels, "status": "complete"}), flush=True)
    return report


def compile_comparison(output: Path, reports: list[dict]) -> dict:
    reference = next(row for row in reports if row["contact_panels"] == 1024)
    metric_keys = (
        "mean_contact_pressure_mpa", "mean_contact_ur_mm", "mean_void_ur_mm",
        "seal_mises_p99_mpa", "seal_mises_max_exclude5deg_mpa",
        "seal_mises_max_exclude10deg_mpa", "concrete_max_principal_p99_mpa",
        "rock_plastic_area_gt_1e6_m2", "rock_plastic_proxy_max",
        "rock_plastic_outer_radius_m",
    )
    rows = []
    for report in sorted(reports, key=lambda item: item["contact_panels"]):
        row = dict(report)
        for key in metric_keys:
            denominator = reference[key]
            row[f"{key}_relative_to_1024_percent"] = (
                100.0 * (report[key] / denominator - 1.0)
                if abs(denominator) > 1.0e-30 else np.nan
            )
        rows.append(row)
    with (output / "panel_convergence_9MPa.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparison = {
        "complete": True,
        "reference_panels": 1024,
        "controlled_parameters": {
            "void_angle_deg": ANGLE_DEG,
            "pressure_mpa": TARGET_PRESSURE_MPA,
            "pressure_increment_mpa": 0.1,
            "rock_mesh": "64x512",
            "core_order": CORE_ORDER,
            "maximum_order": MAXIMUM_ORDER,
            "wall_order": WALL_ORDER,
        },
        "cases": rows,
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return comparison


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "results" / "void_10deg_panel_convergence_9MPa"
    output.mkdir(parents=True, exist_ok=True)
    rock = FullCircleRockDomain(dict(np.load(
        root / "results" / "elastic_rock_kernel_nr64_nt512" / "kernel.npz"
    )))
    saved = np.load(
        root / "results" / "nonaxisymmetric_excavation_nr64_nt512"
        / "installation_state.npz"
    )
    installation = {key: saved[key] for key in ("plastic", "strain", "stress")}
    installation["wall_displacement_m"] = saved["wall_displacement_m"]
    reports = []
    try:
        for panels in PANEL_COUNTS:
            reports.append(run_case(root, rock, installation, panels, output))
        comparison = compile_comparison(output, reports)
        print("COMPLETE", json.dumps(comparison, ensure_ascii=False), flush=True)
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({
            "complete": False, "error": repr(exc),
            "traceback": traceback.format_exc(), "finished_cases": reports,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
