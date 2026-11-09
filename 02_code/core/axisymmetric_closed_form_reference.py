"""Independent axisymmetric reference for nonassociated rock response.

This implements the monotonic closed-form radial solution with an excavation
history and a dilation angle independent of the friction angle.  It is used
only to verify the new radial initial-strain solver under full contact.
Stresses/moduli are in MPa in this file; displacement and radius are in metres.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from axisymmetric_nonassociated_history import lining_axisymmetric_relation


def power_integral(lower: float, upper: float, exponent: float) -> float:
    if upper == lower:
        return 0.0
    t = exponent + 1.0
    log_ratio = math.log(upper / lower)
    if abs(t) < 1.0e-12:
        return log_ratio
    return lower ** t * math.expm1(t * log_ratio) / t


def constants(p: dict) -> dict:
    s = math.sin(math.radians(p["phi_deg"]))
    h = p["cohesion_mpa"] / math.tan(math.radians(p["phi_deg"]))
    pcr0 = p["cohesion_mpa"] * math.cos(math.radians(p["phi_deg"])) / (1.0 - s)
    excavation_plastic = p["p0_mpa"] > pcr0
    rp1 = (
        p["radius_m"]
        * (((p["p0_mpa"] + h) * (1.0 - s) / h) ** ((1.0 - s) / (2.0 * s)))
        if excavation_plastic else p["radius_m"]
    )
    pb1 = 2.0 * pcr0 if excavation_plastic else (
        p["p0_mpa"] * (1.0 + s)
        + p["cohesion_mpa"] * math.cos(math.radians(p["phi_deg"]))
    )
    pb2 = (
        h * (1.0 + s) / (1.0 - s)
        * (rp1 / p["radius_m"]) ** (4.0 * s / (1.0 - s * s)) - h
        if excavation_plastic else None
    )
    return {
        "sin_phi": s,
        "h_mpa": h,
        "pcr0_mpa": pcr0,
        "excavation_plastic": excavation_plastic,
        "excavation_plastic_radius_m": rp1,
        "first_reverse_yield_wall_pressure_mpa": pb1,
        "wall_pressure_for_front_to_reach_excavation_radius_mpa": pb2,
    }


def response(p: dict, wall_pressure_mpa: float) -> dict:
    if wall_pressure_mpa < 0.0:
        raise ValueError("Wall pressure must be nonnegative")
    k = constants(p)
    R, E, nu = p["radius_m"], p["rock_modulus_mpa"], p["rock_poisson"]
    q = wall_pressure_mpa
    if q <= k["first_reverse_yield_wall_pressure_mpa"]:
        return {
            "wall_displacement_m": (1.0 + nu) * R * q / E,
            "new_yield_front_radius_m": R,
            "state": "elastic_reloading",
        }
    s, h = k["sin_phi"], k["h_mpa"]
    t = (1.0 - s) / (1.0 + s)
    m = 1.0 - t
    n = 1.0 / t - 1.0
    psi = math.radians(p["dilation_deg"])
    eta = (1.0 - math.sin(psi)) / (1.0 + math.sin(psi))
    cr = nu * (1.0 + eta) - 1.0
    ct = nu * (1.0 + eta) - eta
    rp1 = k["excavation_plastic_radius_m"]
    if k["excavation_plastic"] and q < k["wall_pressure_for_front_to_reach_excavation_radius_mpa"]:
        a = R * ((q + h) / (h / t)) ** ((1.0 - s * s) / (4.0 * s))
        state = "new_yield_within_excavation_region"
    else:
        a = R * ((q + h) / ((p["p0_mpa"] + h) * (1.0 + s))) ** ((1.0 + s) / (2.0 * s))
        state = (
            "new_yield_outside_excavation_region"
            if k["excavation_plastic"] else "yield_after_elastic_excavation"
        )
    x = a / R
    if k["excavation_plastic"]:
        xe = rp1 / R
        B0 = (p["p0_mpa"] * s + p["cohesion_mpa"] * math.cos(math.radians(p["phi_deg"]))) * rp1 ** 2
    else:
        xe = 1.0
        B0 = p["p0_mpa"] * R ** 2
    pr_a = (q + h) * x ** (-m) - h
    initial_pr_a = h * x ** n - h if k["excavation_plastic"] and a < rp1 else p["p0_mpa"] - B0 / a ** 2
    amplitude = a * a * (pr_a - initial_pr_a)
    wa = (1.0 + nu) * amplitude / (E * a)

    def plastic_integral(lower, upper):
        return (
            (cr + t * ct) * (q + h) * power_integral(lower, upper, eta - m)
            - (cr + ct / t) * h * power_integral(lower, upper, eta + n)
        )

    def elastic_initial_integral(lower, upper):
        return (
            (cr + t * ct) * (q + h) * power_integral(lower, upper, eta - m)
            - (cr + ct) * (p["p0_mpa"] + h) * power_integral(lower, upper, eta)
            + (cr - ct) * B0 / R ** 2 * power_integral(lower, upper, eta - 2.0)
        )

    integral = 0.0
    if xe > 1.0:
        integral += plastic_integral(1.0, min(x, xe))
    if x > max(1.0, xe):
        integral += elastic_initial_integral(max(1.0, xe), x)
    wall_u = x ** eta * wa - (1.0 + nu) * R / E * integral
    return {
        "wall_displacement_m": wall_u,
        "new_yield_front_radius_m": a,
        "state": state,
        "eta_psi": eta,
    }


def solve_coupled(p: dict, gas_pressure_mpa: float) -> dict:
    alpha, beta, unused = lining_axisymmetric_relation()

    def residual(q_mpa: float) -> float:
        lining_u = alpha * gas_pressure_mpa * 1.0e6 - beta * q_mpa * 1.0e6
        return lining_u - response(p, q_mpa)["wall_displacement_m"]

    lower, upper = 0.0, max(gas_pressure_mpa, 1.0)
    while residual(upper) > 0.0:
        upper *= 2.0
    for unused in range(90):
        middle = (lower + upper) / 2.0
        if residual(middle) > 0.0:
            lower = middle
        else:
            upper = middle
    q = (lower + upper) / 2.0
    rock = response(p, q)
    return {
        "gas_pressure_mpa": gas_pressure_mpa,
        "interface_pressure_mpa": q,
        "wall_displacement_mm": rock["wall_displacement_m"] * 1000.0,
        **rock,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    numerical_path = root / "results" / "axisymmetric_nonassociated_baseline" / "nr64_extent2.json"
    numerical = json.loads(numerical_path.read_text(encoding="utf-8"))
    rock = {
        "radius_m": 6.5,
        "rock_modulus_mpa": 12000.0,
        "rock_poisson": 0.26,
        "cohesion_mpa": 0.9,
        "phi_deg": 40.0,
        "dilation_deg": 10.0,
        "p0_mpa": 3.75,
    }
    k = constants(rock)
    alpha, beta, relation = lining_axisymmetric_relation()
    qcrit = k["first_reverse_yield_wall_pressure_mpa"]
    ucrit = response(rock, qcrit)["wall_displacement_m"]
    critical_gas = (ucrit + beta * qcrit * 1.0e6) / (alpha * 1.0e6)
    comparisons = []
    for row in numerical["operation_records"]:
        reference = solve_coupled(rock, row["gas_pressure_mpa"])
        comparisons.append({
            "gas_pressure_mpa": row["gas_pressure_mpa"],
            "numerical_interface_pressure_mpa": row["interface_pressure_mpa"],
            "closed_form_interface_pressure_mpa": reference["interface_pressure_mpa"],
            "pressure_relative_error": row["interface_pressure_mpa"] / reference["interface_pressure_mpa"] - 1.0,
            "numerical_wall_displacement_mm": row["wall_displacement_increment_mm"],
            "closed_form_wall_displacement_mm": reference["wall_displacement_mm"],
            "displacement_relative_error": row["wall_displacement_increment_mm"] / reference["wall_displacement_mm"] - 1.0,
            "closed_form_new_yield_front_radius_m": reference["new_yield_front_radius_m"],
            "closed_form_state": reference["state"],
        })
    final_num = numerical["operation_records"][-1]
    final_ref = solve_coupled(rock, 10.0)
    report = {
        "rock_parameters": rock,
        "closed_form_constants": k,
        "lining_relation": relation,
        "closed_form_first_reverse_yield_gas_pressure_mpa": critical_gas,
        "numerical_bracket_for_first_reverse_yield_mpa": numerical["first_operating_plasticity"],
        "maximum_interface_pressure_relative_error": max(abs(v["pressure_relative_error"]) for v in comparisons),
        "maximum_wall_displacement_relative_error": max(abs(v["displacement_relative_error"]) for v in comparisons),
        "final_new_yield_front_radius_relative_error": (
            final_num["plastic_outer_radius_m"] / final_ref["new_yield_front_radius_m"] - 1.0
        ),
        "comparisons": comparisons,
    }
    output = root / "results" / "axisymmetric_nonassociated_baseline" / "closed_form_comparison.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "comparisons"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
