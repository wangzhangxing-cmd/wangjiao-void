"""Recover seal stresses from the finite-thickness layered ring solution."""
from __future__ import annotations

import numpy as np

from layered_ring_operator import SealLiningOperator


def _surface_coefficients(interface, lining_u: np.ndarray,
                          internal_pressure_pa: float) -> dict[str, np.ndarray]:
    """Return inner/outer seal stress Fourier coefficients.

    Components are ordered as rr, theta-theta, zz and r-theta.  The stored
    Fourier convention is the same one used by ``local_void_interface``.
    """
    operator: SealLiningOperator = interface.op.layers
    # The finite-thickness layered operator is evaluated exactly through the
    # interface core order.  Higher interface modes use a regular-disk tail
    # only to stabilise the displacement solution and cannot be consistently
    # back-propagated through all seal/lining layers.
    count = min(len(lining_u), int(interface.op.core_order) + 1)
    result = {
        "inner": np.zeros((count, 4), complex),
        "outer": np.zeros((count, 4), complex),
    }
    material = operator.seal.segments[0].material
    C11, C12, C22, C13, C23, unused_g = material.plane_strain_stiffness()

    for n in range(count):
        mode = operator.mode(n)
        inner_force = np.array([internal_pressure_pa, 0.0], complex) if n == 0 else np.zeros(2, complex)
        joint = mode["H"] @ lining_u[n] + mode["J"] @ inner_force
        u_inner, u_outer = joint[:2], joint[2:]
        unused_radii, pieces, unused_prefix, unused_materials = operator.seal.chain(n)
        displacement = operator.seal.internal_displacements(n, u_inner, u_outer)

        first = pieces[0]
        external_inner = first[:2, :2] @ displacement[0] + first[:2, 2:] @ displacement[1]
        last = pieces[-1]
        external_outer = last[2:, :2] @ displacement[-2] + last[2:, 2:] @ displacement[-1]

        for name, radius, u, traction, sign in (
            ("inner", operator.seal.inner_radius_m, u_inner, external_inner, -1.0),
            ("outer", operator.seal.outer_radius_m, u_outer, external_outer, 1.0),
        ):
            sigma_rr = sign * traction[0]
            sigma_rt = sign * traction[1]
            epsilon_tt = (u[0] + 1j * n * u[1]) / radius
            epsilon_rr = (sigma_rr - C12 * epsilon_tt) / C11
            sigma_tt = C12 * epsilon_rr + C22 * epsilon_tt
            sigma_zz = C13 * epsilon_rr + C23 * epsilon_tt
            result[name][n] = (sigma_rr, sigma_tt, sigma_zz, sigma_rt)
    return result


def recover_seal_surface_displacement(interface, lining_u: np.ndarray,
                                      internal_pressure_pa: float,
                                      theta: np.ndarray) -> dict[str, np.ndarray]:
    """Return inner and outer seal-surface displacements in polar components."""
    operator: SealLiningOperator = interface.op.layers
    count = min(len(lining_u), int(interface.op.core_order) + 1)
    coefficients = {
        "inner": np.zeros((count, 2), complex),
        "outer": np.zeros((count, 2), complex),
    }
    for n in range(count):
        mode = operator.mode(n)
        inner_force = np.array([internal_pressure_pa, 0.0], complex) if n == 0 else np.zeros(2, complex)
        joint = mode["H"] @ lining_u[n] + mode["J"] @ inner_force
        coefficients["inner"][n] = joint[:2]
        coefficients["outer"][n] = joint[2:]
    return {name: reconstruct_real(values, theta) for name, values in coefficients.items()}


def reconstruct_real(coefficients: np.ndarray, theta: np.ndarray,
                     chunk: int = 512) -> np.ndarray:
    theta = np.asarray(theta, float)
    output = np.tile(coefficients[0].real, (len(theta), 1))
    for first in range(1, len(coefficients), chunk):
        n = np.arange(first, min(first + chunk, len(coefficients)))
        phase = np.exp(1j * theta[:, None] * n[None, :])
        output += np.real(phase @ coefficients[n])
    return output


def von_mises(stress: np.ndarray) -> np.ndarray:
    rr, tt, zz, rt = (stress[:, i] for i in range(4))
    return np.sqrt(0.5 * ((rr - tt) ** 2 + (tt - zz) ** 2 + (zz - rr) ** 2) + 3.0 * rt ** 2)


def maximum_principal(stress: np.ndarray) -> np.ndarray:
    """Maximum principal stress for rr, tt, zz and rt components."""
    rr, tt, zz, rt = (stress[:, i] for i in range(4))
    in_plane = 0.5 * (rr + tt) + np.sqrt((0.5 * (rr - tt)) ** 2 + rt ** 2)
    return np.maximum(in_plane, zz)


def recover_seal_stress(interface, lining_u: np.ndarray,
                        internal_pressure_pa: float,
                        theta: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    coefficients = _surface_coefficients(interface, lining_u, internal_pressure_pa)
    output = {}
    for surface, values in coefficients.items():
        stress = reconstruct_real(values, theta)
        output[surface] = {"stress": stress, "mises": von_mises(stress)}
    return output


def recover_concrete_face_stress(interface, lining_u: np.ndarray,
                                 internal_pressure_pa: float,
                                 theta: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Recover stresses on the inner and outer faces of every lining layer."""
    operator: SealLiningOperator = interface.op.layers
    count = min(len(lining_u), int(interface.op.core_order) + 1)
    faces = []
    for index, segment in enumerate(operator.lining.segments):
        faces.extend([
            (f"layer{index + 1:02d}_inner", index, "inner", segment.inner_radius_m),
            (f"layer{index + 1:02d}_outer", index, "outer", segment.outer_radius_m),
        ])
    coefficients = {name: np.zeros((count, 4), complex) for name, *_ in faces}

    for n in range(count):
        mode = operator.mode(n)
        inner_force = np.array([internal_pressure_pa, 0.0], complex) if n == 0 else np.zeros(2, complex)
        joint = mode["H"] @ lining_u[n] + mode["J"] @ inner_force
        lining_inner = joint[2:]
        radii, pieces, unused_prefix, unused_materials = operator.lining.chain(n)
        displacement = operator.lining.internal_displacements(n, lining_inner, lining_u[n])

        for name, layer_index, side, radius in faces:
            nodes = np.flatnonzero(np.isclose(radii, radius, rtol=0.0, atol=1.0e-11))
            if len(nodes) != 1:
                raise RuntimeError(f"Lining layer radius {radius} not found uniquely")
            node = int(nodes[0])
            segment = operator.lining.segments[layer_index]
            material = segment.material
            C11, C12, C22, C13, C23, unused_g = material.plane_strain_stiffness()
            if side == "inner":
                piece = pieces[node]
                traction = piece[:2, :2] @ displacement[node] + piece[:2, 2:] @ displacement[node + 1]
                sign = -1.0
            else:
                piece = pieces[node - 1]
                traction = piece[2:, :2] @ displacement[node - 1] + piece[2:, 2:] @ displacement[node]
                sign = 1.0
            u = displacement[node]
            sigma_rr = sign * traction[0]
            sigma_rt = sign * traction[1]
            epsilon_tt = (u[0] + 1j * n * u[1]) / radius
            epsilon_rr = (sigma_rr - C12 * epsilon_tt) / C11
            sigma_tt = C12 * epsilon_rr + C22 * epsilon_tt
            sigma_zz = C13 * epsilon_rr + C23 * epsilon_tt
            coefficients[name][n] = (sigma_rr, sigma_tt, sigma_zz, sigma_rt)

    output = {}
    for name, layer_index, side, radius in faces:
        stress = reconstruct_real(coefficients[name], theta)
        output[name] = {
            "stress": stress,
            "maximum_principal": maximum_principal(stress),
            "radius_m": np.array(radius),
            "layer_index": np.array(layer_index + 1),
        }
    return output


def recover_lining_section_response(interface, lining_u: np.ndarray,
                                    internal_pressure_pa: float,
                                    theta: np.ndarray) -> dict[str, np.ndarray]:
    """Recover lining hoop resultants and physical reinforcement stresses.

    The hoop stress is integrated through every radial layer with a linear
    interpolation between its two faces.  Equivalent reinforcement bands are
    used only to obtain the section deformation; physical bar stress is
    recovered from the band hoop strain multiplied by the reinforcing-steel
    modulus.
    """
    operator: SealLiningOperator = interface.op.layers
    count = min(len(lining_u), int(interface.op.core_order) + 1)
    segments = operator.lining.segments
    face_tt = np.zeros((len(segments), 2, count), complex)
    face_ett = np.zeros_like(face_tt)

    for n in range(count):
        mode = operator.mode(n)
        inner_force = np.array([internal_pressure_pa, 0.0], complex) if n == 0 else np.zeros(2, complex)
        joint = mode["H"] @ lining_u[n] + mode["J"] @ inner_force
        lining_inner = joint[2:]
        radii, pieces, unused_prefix, unused_materials = operator.lining.chain(n)
        displacement = operator.lining.internal_displacements(n, lining_inner, lining_u[n])

        for index, segment in enumerate(segments):
            material = segment.material
            C11, C12, C22 = material.plane_strain_stiffness()[:3]
            for side_index, (radius, side) in enumerate((
                    (segment.inner_radius_m, "inner"),
                    (segment.outer_radius_m, "outer"))):
                nodes = np.flatnonzero(np.isclose(radii, radius, rtol=0.0, atol=1.0e-11))
                if len(nodes) != 1:
                    raise RuntimeError(f"Lining layer radius {radius} not found uniquely")
                node = int(nodes[0])
                if side == "inner":
                    piece = pieces[node]
                    traction = piece[:2, :2] @ displacement[node] + piece[:2, 2:] @ displacement[node + 1]
                    sign = -1.0
                else:
                    piece = pieces[node - 1]
                    traction = piece[2:, :2] @ displacement[node - 1] + piece[2:, 2:] @ displacement[node]
                    sign = 1.0
                u = displacement[node]
                sigma_rr = sign * traction[0]
                epsilon_tt = (u[0] + 1j * n * u[1]) / radius
                epsilon_rr = (sigma_rr - C12 * epsilon_tt) / C11
                face_tt[index, side_index, n] = C12 * epsilon_rr + C22 * epsilon_tt
                face_ett[index, side_index, n] = epsilon_tt

    section_mid = 0.5 * (
        operator.lining_input.inner_radius_m + operator.lining_input.outer_radius_m
    )
    force = np.zeros(len(theta))
    moment = np.zeros(len(theta))
    rebar = {
        "inner": np.full(len(theta), np.nan),
        "outer": np.full(len(theta), np.nan),
    }
    for index, segment in enumerate(segments):
        s0 = reconstruct_real(face_tt[index, 0, :, None], theta)[:, 0]
        s1 = reconstruct_real(face_tt[index, 1, :, None], theta)[:, 0]
        r0, r1 = segment.inner_radius_m, segment.outer_radius_m
        h = r1 - r0
        z0, z1 = r0 - section_mid, r1 - section_mid
        force += 0.5 * h * (s0 + s1)
        moment += h / 6.0 * ((2.0 * z0 + z1) * s0 + (z0 + 2.0 * z1) * s1)
        label = segment.material.name
        for key in ("inner", "outer"):
            if f"{key}_reinforcement" in label:
                e0 = reconstruct_real(face_ett[index, 0, :, None], theta)[:, 0]
                e1 = reconstruct_real(face_ett[index, 1, :, None], theta)[:, 0]
                rebar[key] = 0.5 * (e0 + e1) * operator.lining_input.steel_modulus_pa

    return {
        "lining_N": force,
        "lining_M": moment,
        "rebar_in": rebar["inner"],
        "rebar_out": rebar["outer"],
    }
