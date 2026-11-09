"""Section stiffness of the cracked, doubly reinforced concrete lining.

The calculation is per metre of cavern axis.  Circumferential reinforcement
is treated as discrete circular bars at its actual radial centroid.  Residual
tension stiffening of cracked concrete is represented by alpha*Ec.  Steel
replaces, rather than overlaps, the residual concrete occupying the bar area.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class BarLayer:
    name: str
    diameter_m: float
    spacing_m: float
    centroid_from_inner_face_m: float

    @property
    def area_per_m(self) -> float:
        return math.pi * self.diameter_m ** 2 / (4.0 * self.spacing_m)

    @property
    def centroidal_second_moment_per_m(self) -> float:
        return math.pi * self.diameter_m ** 4 / (64.0 * self.spacing_m)

    @property
    def equal_second_moment_band_thickness_m(self) -> float:
        # A uniform band with this thickness has I/A=t^2/12=d^2/16,
        # identical to a circular bar about its centroidal tangential axis.
        return math.sqrt(3.0 / 4.0) * self.diameter_m


@dataclass(frozen=True)
class SectionInput:
    thickness_m: float = 0.50
    cover_m: float = 0.050
    concrete_modulus_pa: float = 33.4e9
    steel_modulus_pa: float = 200.0e9
    inner_diameter_m: float = 0.012
    inner_spacing_m: float = 0.200
    outer_diameter_m: float = 0.020
    outer_spacing_m: float = 0.200

    def bar_layers(self) -> tuple[BarLayer, BarLayer]:
        return (
            BarLayer(
                "inner",
                self.inner_diameter_m,
                self.inner_spacing_m,
                self.cover_m + self.inner_diameter_m / 2.0,
            ),
            BarLayer(
                "outer",
                self.outer_diameter_m,
                self.outer_spacing_m,
                self.thickness_m - self.cover_m - self.outer_diameter_m / 2.0,
            ),
        )


def calculate(section: SectionInput, residual_ratio: float) -> dict:
    if not 0.0 <= residual_ratio <= 1.0:
        raise ValueError("residual_ratio must lie in [0, 1]")
    h, Ec, Es = section.thickness_m, section.concrete_modulus_pa, section.steel_modulus_pa
    y_mid = h / 2.0
    bars = section.bar_layers()
    A = residual_ratio * Ec * h
    B = 0.0
    D = residual_ratio * Ec * h ** 3 / 12.0
    rows = []
    for bar in bars:
        y = bar.centroid_from_inner_face_m - y_mid
        replacement_modulus = Es - residual_ratio * Ec
        Ai = replacement_modulus * bar.area_per_m
        Bi = Ai * y
        Di = replacement_modulus * (
            bar.area_per_m * y ** 2 + bar.centroidal_second_moment_per_m
        )
        A += Ai
        B += Bi
        D += Di
        rows.append(
            {
                **asdict(bar),
                "area_mm2_per_m": bar.area_per_m * 1.0e6,
                "y_from_middepth_m": y,
                "equal_second_moment_band_thickness_m": bar.equal_second_moment_band_thickness_m,
                "A_N_per_m": Ai,
                "B_N": Bi,
                "D_Nm": Di,
            }
        )
    gross_A = Ec * h
    gross_D = Ec * h ** 3 / 12.0
    D_zero_N = D - B * B / A
    total_steel_area = sum(v.area_per_m for v in bars)
    return {
        "residual_concrete_hoop_ratio": residual_ratio,
        "reinforcement_ratio": total_steel_area / h,
        "total_steel_area_mm2_per_m": total_steel_area * 1.0e6,
        "A_N_per_m": A,
        "B_N": B,
        "D_Nm": D,
        "D_at_zero_membrane_force_Nm": D_zero_N,
        "eta_A": A / gross_A,
        "eta_B": B / (Ec * h ** 2),
        "eta_D": D / gross_D,
        "eta_D_at_zero_membrane_force": D_zero_N / gross_D,
        "bar_layers": rows,
    }


def main() -> None:
    section = SectionInput()
    result = {
        "section_input": asdict(section),
        "cases": [calculate(section, alpha) for alpha in (0.0, 0.05, 0.10)],
        "interpretation": (
            "alpha=0 is the fully cracked steel-only limit; alpha=0.05 is the "
            "baseline residual tension-stiffening assumption; alpha=0.10 is the upper sensitivity case."
        ),
    }
    root = Path(__file__).resolve().parents[2]
    output = root / "results" / "reinforced_concrete_section_properties.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
