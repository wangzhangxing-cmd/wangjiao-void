"""Generate a fresh nonassociated 32x256 excavation history."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import numpy as np
import argparse

from nonaxisymmetric_nonassociated_domain import FullCircleRockDomain


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nr", type=int, default=32)
    parser.add_argument("--nt", type=int, default=256)
    arguments = parser.parse_args()
    began = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    label = f"nr{arguments.nr}_nt{arguments.nt}"
    kernel_path = root / "results" / f"elastic_rock_kernel_{label}" / "kernel.npz"
    radial_path = (
        root / "results" / "axisymmetric_nonassociated_baseline"
        / f"nr{arguments.nr}_extent2_states.npz"
    )
    output = root / "results" / f"nonaxisymmetric_excavation_{label}"
    output.mkdir(parents=True, exist_ok=True)
    kernel = dict(np.load(kernel_path))
    domain = FullCircleRockDomain(kernel)
    state = domain.initial_state()
    records = []
    for index, pressure in enumerate(np.linspace(domain.p0, 0.0, 31)[1:], 1):
        state = domain.step(state, float(pressure))
        local_stress = domain.local_components(state["stress"])
        local_plastic = domain.local_components(state["plastic"])
        local_strain = domain.local_components(state["strain"])
        row = {
            "step": index,
            "wall_pressure_mpa": pressure / 1.0e6,
            "wall_displacement_mm": state["wall_displacement_m"] * 1000.0,
            "iterations": state["iterations"],
            "source_gap": state["source_gap"],
            "stress_mismatch_pa": state["stress_mismatch_pa"],
            "yield_residual_pa": state["yield_residual_pa"],
            "local_stress_angular_range_pa": float(np.max(np.ptp(local_stress, axis=1))),
            "local_plastic_angular_range": float(np.max(np.ptp(local_plastic, axis=1))),
            "local_shear_stress_max_pa": float(np.max(np.abs(local_stress[:, :, 3]))),
            "total_axial_strain_max": float(np.max(np.abs(local_strain[:, :, 2]))),
            "outer_layer_plastic_max": float(np.max(np.abs(local_plastic[-1]))),
        }
        records.append(row)
        print(json.dumps(row), flush=True)

    radial = np.load(radial_path)
    expected_stress = radial["installation_stress"]
    expected_plastic = radial["installation_plastic"]
    local_stress = domain.local_components(state["stress"])
    local_plastic = domain.local_components(state["plastic"])
    stress_gap = float(np.max(np.abs(local_stress[:, :, :3] - expected_stress[:, None, :])))
    plastic_gap = float(np.max(np.abs(local_plastic[:, :, :3] - expected_plastic[:, None, :])))
    wall_gap = state["wall_displacement_m"] - float(radial["installation_wall_displacement_m"])
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("nonaxisymmetric_nonassociated_domain.py"),
        Path(__file__).with_name("batch_nonassociated.py"),
        Path(__file__).with_name("mohr_coulomb_nonassociated.py"),
        kernel_path,
    ]
    signature = hashlib.sha256(
        json.dumps({str(p.relative_to(root)): sha256(p) for p in source_paths}, sort_keys=True).encode()
    ).hexdigest()
    np.savez_compressed(
        output / "installation_state.npz",
        plastic=state["plastic"], strain=state["strain"], stress=state["stress"],
        wall_displacement_m=np.array(state["wall_displacement_m"]),
        wall_pressure_pa=np.array(state["wall_pressure_pa"]),
        state_codes=state["state_codes"], radii=kernel["radii"], angles=kernel["angles"],
        signature=np.array(signature),
    )
    report = {
        "complete": True,
        "signature": signature,
        "nr": domain.nr,
        "nt": domain.nt,
        "material_points": domain.n,
        "loading": "3.75 MPa wall pressure unloaded to zero in 30 equal steps",
        "flow_rule": {"friction_angle_deg": 40.0, "dilation_angle_deg": 10.0},
        "records": records,
        "final_checks": {
            "wall_displacement_mm": state["wall_displacement_m"] * 1000.0,
            "axisymmetric_wall_displacement_difference_mm": wall_gap * 1000.0,
            "axisymmetric_stress_max_difference_pa": stress_gap,
            "axisymmetric_plastic_max_difference": plastic_gap,
            "maximum_angular_stress_range_pa": max(v["local_stress_angular_range_pa"] for v in records),
            "maximum_angular_plastic_range": max(v["local_plastic_angular_range"] for v in records),
            "maximum_total_axial_strain": max(v["total_axial_strain_max"] for v in records),
            "outer_layer_plastic_max": max(v["outer_layer_plastic_max"] for v in records),
            "maximum_iterations": max(v["iterations"] for v in records),
            "maximum_yield_residual_pa": max(v["yield_residual_pa"] for v in records),
        },
        "kernel_provenance": str(
            root / "results" / f"elastic_rock_kernel_{label}" / "manifest.json"
        ),
        "elapsed_seconds": time.perf_counter() - began,
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("COMPLETE", json.dumps(report["final_checks"]), flush=True)


if __name__ == "__main__":
    main()
