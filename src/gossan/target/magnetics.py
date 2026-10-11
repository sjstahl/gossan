"""Features from a regional magnetic grid: the field and what its derivatives say about the rocks.

The raw total magnetic field is a poor model input on its own: its absolute level depends on the
survey, and a high or a low can mean either a magnetic source or its dipole's other pole. Derivatives
turn it into maps of *sources and edges*, which is what a geological model can use:

``mag_field``    the grid itself (nT), as supplied;
``mag_local``    the field minus its upward continuation to ``regional_m``: near-surface anomalies
                 with the deep, long-wavelength trend removed;
``mag_vd1``      first vertical derivative (nT/km), taken downward toward the sources as is usual, so it
                 is positive over a positive anomaly: sharpens anomalies and separates close sources;
``mag_as``       analytic signal amplitude (nT/km): peaks over magnetic sources whatever the
                 direction of magnetization;
``mag_tilt``     tilt angle (degrees): arctangent of the vertical over the total horizontal derivative;
                 about +90 over a source top, 0 over its edges, and independent of amplitude, so it
                 shows weak and strong sources alike (contacts, faults, intrusions);
``mag_texture``  standard deviation within ``texture_m``: how disturbed the field is locally.

Derivatives are computed in the wavenumber domain (FFT) on the grid's own pixels, reflecting the
edges so the transform does not wrap, and nodata is filled by nearest valid values before and
restored after. The grid must be in a projected CRS in metres (``gossan ingest`` puts it in the
project CRS). Edges of the grid are less reliable: ingest with a buffer
(``gossan ingest ... --buffer-m 5000``) so the area of interest is well inside.

Limits: no reduction to the pole is applied. At BC latitudes (inclination about 72 to 76 degrees) the
dipole asymmetry is modest, and the analytic signal and the tilt angle are the least sensitive to it;
``mag_field``, ``mag_local`` and ``mag_vd1`` are not. The numbers come from a regional compilation
(200 m or 100 m cells), not a detailed survey: they say where the magnetic ground changes over a
few hundred metres, not what a claim-scale survey would show.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

NAMES = ("mag_field", "mag_local", "mag_vd1", "mag_as", "mag_tilt", "mag_texture")
#: Candidate names of the ingested magnetic raster (``rasters/<name>.tif``), tried in this order.
RASTER_NAMES = ("magnetics", "nrcan_magnetics", "mag", "tmi", "rtf", "rmi")
M_TO_KM = 1000.0


class MagneticsError(Exception):
    """The grid cannot be used; the message says why."""


@dataclass
class Derivatives:
    """Derived layers on the input grid (NaN where the input had no data)."""

    layers: dict[str, np.ndarray]
    dx: float


def _wavenumbers(shape: tuple[int, int], dx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ky = 2 * np.pi * np.fft.fftfreq(shape[0], d=dx)
    kx = 2 * np.pi * np.fft.fftfreq(shape[1], d=dx)
    kxx, kyy = np.meshgrid(kx, ky)
    return kxx, kyy, np.hypot(kxx, kyy)


def _fill_nodata(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``field`` with every invalid cell replaced by the nearest valid one."""
    if valid.all():
        return field
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return field[tuple(idx)]


def pad_grid(field: np.ndarray, pad: int) -> np.ndarray:
    """``field`` extended by ``pad`` cells on every side: reflected, then faded to the mean over the
    extension with a cosine taper, so the periodic transform sees no jump at the padded edge."""
    if not pad:
        return field
    padded = np.pad(field, pad, mode="reflect")
    ramp = 0.5 * (1.0 - np.cos(np.pi * (np.arange(pad) + 0.5) / pad))
    wy = np.ones(padded.shape[0])
    wy[:pad], wy[-pad:] = ramp, ramp[::-1]
    wx = np.ones(padded.shape[1])
    wx[:pad], wx[-pad:] = ramp, ramp[::-1]
    mean = float(field.mean())
    return mean + (padded - mean) * np.outer(wy, wx)


def _spectral(field: np.ndarray, operator: np.ndarray, pad: int) -> np.ndarray:
    """Apply a wavenumber-domain ``operator`` (shaped for the padded grid) and crop back."""
    out = np.real(np.fft.ifft2(np.fft.fft2(pad_grid(field, pad)) * operator))
    return out[pad : out.shape[0] - pad, pad : out.shape[1] - pad] if pad else out


def _pad_for(shape: tuple[int, int], pad: int | None) -> int:
    """The padding in cells: ``pad`` if given, else a quarter of the larger side (at least 16)."""
    if pad is not None:
        return int(pad)
    return int(min(max(16, 0.25 * max(shape)), min(shape) - 1))


def upward_continue(
    field: np.ndarray, dx: float, height_m: float, valid: np.ndarray | None = None, *, pad: int | None = None
) -> np.ndarray:
    """``field`` continued upward by ``height_m`` metres (smoothing it: deep and regional sources stay)."""
    valid = np.isfinite(field) if valid is None else valid
    filled = _fill_nodata(np.where(valid, field, 0.0), valid)
    pad = _pad_for(field.shape, pad)
    _, _, k = _wavenumbers((field.shape[0] + 2 * pad, field.shape[1] + 2 * pad), dx)
    out = _spectral(filled, np.exp(-k * height_m), pad)
    return np.where(valid, out, np.nan)


def derivatives(
    field: np.ndarray, dx: float, *, regional_m: float = 2000.0, texture_m: float = 1000.0, pad: int | None = None
) -> Derivatives:
    """The six layers named in ``NAMES`` from a magnetic ``field`` grid with square cells of ``dx`` metres.

    ``field`` has NaN where there is no data; ``pad`` overrides the edge padding in cells (0: none,
    exact only for a field periodic across the grid). Raises ``MagneticsError`` for a grid too small to
    differentiate or without any data.
    """
    if field.ndim != 2 or min(field.shape) < 8:
        raise MagneticsError(f"the magnetic grid is {field.shape[1]} x {field.shape[0]} cells: too small to take derivatives")
    valid = np.isfinite(field)
    if not valid.any():
        raise MagneticsError("the magnetic grid has no data inside the area")
    filled = _fill_nodata(np.where(valid, field, 0.0).astype(np.float64), valid)
    pad = _pad_for(field.shape, pad)
    kx, ky, k = _wavenumbers((field.shape[0] + 2 * pad, field.shape[1] + 2 * pad), dx)
    d_x = _spectral(filled, 1j * kx, pad)
    d_y = _spectral(filled, 1j * ky, pad)
    d_z = _spectral(filled, k, pad)  # vertical derivative of a potential field: |k| times its transform
    thd = np.hypot(d_x, d_y)
    regional = _spectral(filled, np.exp(-k * regional_m), pad)
    size = max(3, round(texture_m / dx) | 1)
    mean = uniform_filter(filled, size=size, mode="reflect")
    texture = np.sqrt(np.maximum(uniform_filter(filled**2, size=size, mode="reflect") - mean**2, 0.0))
    layers = {
        "mag_field": field.astype(np.float64),
        "mag_local": filled - regional,
        "mag_vd1": d_z * M_TO_KM,
        "mag_as": np.sqrt(d_x**2 + d_y**2 + d_z**2) * M_TO_KM,
        "mag_tilt": np.degrees(np.arctan2(d_z, thd)),
        "mag_texture": texture,
    }
    return Derivatives({name: np.where(valid, arr, np.nan) for name, arr in layers.items()}, dx)
