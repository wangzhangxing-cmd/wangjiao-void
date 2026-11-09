"""Exact cell-average elastic response to a balanced wall-traction series."""
from __future__ import annotations

import numpy as np


def radial_power_mean(a: np.ndarray, b: np.ndarray, powers: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, float)[:, None], np.asarray(b, float)[:, None]
    exponent = np.asarray(powers, float)[None] + 2.0
    safe = np.where(exponent == 0.0, 1.0, exponent)
    value = (
        2.0 * np.exp(safe * np.log(a)) * np.expm1(safe * np.log(b / a))
        / (safe * (b * b - a * a))
    )
    return np.where(exponent == 0.0, 2.0 * np.log(b / a) / (b * b - a * a), value)


def stress_to_strain(stress: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Plane-strain elastic strain [xx, yy, zz, xy], tensor shear."""
    stress = np.asarray(stress)
    strain = np.zeros_like(stress)
    factor = (1.0 + nu) / E
    strain[..., 0] = factor * ((1.0 - nu) * stress[..., 0] - nu * stress[..., 1])
    strain[..., 1] = factor * ((1.0 - nu) * stress[..., 1] - nu * stress[..., 0])
    strain[..., 3] = factor * stress[..., 3]
    return strain


def sector_average_stress(traction: np.ndarray, radii: np.ndarray, angular_cells: int,
                          radius: float = 6.5, nu: float = 0.26) -> np.ndarray:
    """Return exact annular-sector averages in fixed Cartesian components."""
    traction = np.asarray(traction, complex)
    radii = np.asarray(radii, float)
    nt = int(angular_cells)
    if (traction.ndim != 2 or traction.shape[1] != 2 or len(traction) == 0 or
            not np.all(np.isfinite(traction)) or nt < 4 or len(radii) < 2 or
            np.any(np.diff(radii) <= 0.0) or radii[0] < radius - 1.0e-12):
        raise ValueError("Invalid traction series or target grid")
    scale = max(float(np.max(np.abs(traction))), 1.0)
    if np.max(np.abs(traction[0].imag)) > 1.0e-12 * scale:
        raise ValueError("Mean traction must be real")
    if len(traction) > 1 and abs(traction[1, 1] + 1j * traction[1, 0]) > 1.0e-12 * scale:
        raise ValueError("The actual wall traction has nonzero resultant")
    a, b = radii[:-1] / radius, radii[1:] / radius
    nr, width = len(a), 2.0 * np.pi / nt
    bins = np.zeros((nr, nt, 4), complex)
    band = 4 * nt
    for first in range(0, len(traction), band):
        orders = np.arange(first, min(first + band, len(traction)))
        count = len(orders)
        raw = np.zeros((nr, count, 4), complex)
        positive = orders >= 2
        if np.any(positive):
            n = orders[positive].astype(float)
            tr, tt = traction[orders[positive], 0], traction[orders[positive], 1]
            p = radial_power_mean(a, b, -n)
            q = radial_power_mean(a, b, -n - 2.0)
            x, y = tr - 1j * tt, n * tr - (n + 2.0) * 1j * tt
            raw[:, positive, 0] = (-(n + 2.0) * x * p + y * q) / 2.0
            raw[:, positive, 1] = ((n - 2.0) * x * p - y * q) / 2.0
            raw[:, positive, 2] = -2.0 * nu * x * p
            raw[:, positive, 3] = 1j * (n * x * p - y * q) / 2.0
        if first == 0:
            inverse_square = radial_power_mean(a, b, np.array([-2.0]))[:, 0]
            raw[:, 0, 0] = -traction[0, 0] * inverse_square
            raw[:, 0, 1] = traction[0, 0] * inverse_square
            raw[:, 0, 3] = -traction[0, 1] * inverse_square
            if len(traction) > 1:
                inverse_cube = radial_power_mean(a, b, np.array([-3.0]))[:, 0]
                raw[:, 1, 0] = -traction[1, 0] * inverse_cube
                raw[:, 1, 1] = traction[1, 0] * inverse_cube
                raw[:, 1, 3] = 1j * traction[1, 0] * inverse_cube
        mean, deviator, shear = (
            (raw[..., 0] + raw[..., 1]) / 2.0,
            (raw[..., 0] - raw[..., 1]) / 2.0,
            raw[..., 3],
        )
        moment = lambda k: np.sinc(k * width / (2.0 * np.pi))
        mn = moment(orders)
        cosine = (moment(orders + 2) + moment(orders - 2)) / 2.0
        sine = (moment(orders + 2) - moment(orders - 2)) / (2j)
        fixed = np.stack([
            mean * mn + deviator * cosine - shear * sine,
            mean * mn - deviator * cosine + shear * sine,
            raw[..., 2] * mn,
            deviator * sine + shear * cosine,
        ], axis=-1)
        fixed *= np.exp(1j * orders * width / 2.0)[None, :, None]
        padded = np.zeros((nr, band, 4), complex)
        padded[:, :count] = fixed
        bins += padded.reshape(nr, 4, nt, 4).sum(axis=1)
    local = np.real(np.fft.ifft(bins, axis=1) * nt)
    theta = (np.arange(nt) + 0.5) * width
    cosine, sine = np.cos(theta)[None, :], np.sin(theta)[None, :]
    rr, tt, zz, rt = (local[..., component] for component in range(4))
    return np.stack([
        cosine * cosine * rr + sine * sine * tt - 2.0 * cosine * sine * rt,
        sine * sine * rr + cosine * cosine * tt + 2.0 * cosine * sine * rt,
        zz,
        cosine * sine * (rr - tt) + (cosine * cosine - sine * sine) * rt,
    ], axis=-1)


def sector_average_strain(traction: np.ndarray, radii: np.ndarray, angular_cells: int,
                          E: float = 12.0e9, nu: float = 0.26,
                          radius: float = 6.5) -> np.ndarray:
    return stress_to_strain(
        sector_average_stress(traction, radii, angular_cells, radius, nu), E, nu
    )

