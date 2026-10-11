"""Magnetic derivatives: checked against fields whose derivatives are known exactly."""

from __future__ import annotations

import numpy as np
import pytest

from gossan.target.magnetics import NAMES, MagneticsError, derivatives, upward_continue

DX = 100.0
N = 128


def grid_xy(n: int = N, dx: float = DX) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(n) + 0.5) * dx
    return np.meshgrid(x, x)


def plane_wave(cycles: int, amplitude: float = 50.0) -> tuple[np.ndarray, float]:
    """cos(k x) with a whole number of wavelengths across the grid: periodic, so the transform is exact
    without padding (``pad=0``)."""
    xx, _ = grid_xy()
    k = 2 * np.pi * cycles / (N * DX)
    return amplitude * np.cos(k * (xx - DX / 2)), k


NODD = 129  # an odd grid has a cell exactly at its centre


def dipole(depth: float = 400.0, amplitude: float = 1e9) -> np.ndarray:
    """The vertical field of a vertical dipole buried ``depth`` metres below the centre cell of an odd
    grid: a potential field, positive over the source, with a negative collar."""
    xx, yy = grid_xy(NODD)
    centre = (NODD // 2 + 0.5) * DX
    r2 = (xx - centre) ** 2 + (yy - centre) ** 2
    return amplitude * (2 * depth**2 - r2) / (r2 + depth**2) ** 2.5


# --- the transforms ---------------------------------------------------------------------------------------------


def test_vertical_derivative_of_a_wave_is_its_wavenumber_times_the_field():
    field, k = plane_wave(cycles=8)
    got = derivatives(field, DX, pad=0).layers["mag_vd1"]
    assert np.allclose(got, k * field * 1000.0, rtol=1e-9, atol=1e-9)  # downward derivative |k| F, in nT/km: exact


def test_with_the_default_padding_the_interior_is_still_accurate():
    field, k = plane_wave(cycles=8)
    got = derivatives(field, DX).layers["mag_vd1"]  # not periodic across the padded grid: edges are approximate
    expected = k * field * 1000.0
    inner = (slice(16, -16), slice(16, -16))
    assert np.allclose(got[inner], expected[inner], atol=0.03 * np.abs(expected).max())


def test_upward_continuation_damps_a_wave_by_exp_minus_k_h():
    field, k = plane_wave(cycles=8)
    for h in (200.0, 1000.0):
        assert np.allclose(upward_continue(field, DX, h, pad=0), field * np.exp(-k * h), atol=1e-9), h
    assert np.allclose(upward_continue(field, DX, 0.0, pad=0), field, atol=1e-9)


def test_local_field_is_what_upward_continuation_leaves_out():
    for cycles in (1, 24):
        wave, k = plane_wave(cycles=cycles, amplitude=100.0)
        local = derivatives(wave, DX, regional_m=2000.0, pad=0).layers["mag_local"]
        assert np.allclose(local, wave * (1 - np.exp(-k * 2000.0)), atol=1e-9)  # exactly the wave minus its continued part
    long_ratio = 1 - np.exp(-2 * np.pi / (N * DX) * 2000.0)  # a 12.8 km wave: only 62% survives a 2 km continuation
    short_ratio = 1 - np.exp(-2 * np.pi * 24 / (N * DX) * 2000.0)
    assert long_ratio == pytest.approx(0.625, abs=0.01) and short_ratio > 0.999  # a short wave is all local


def test_a_source_peaks_the_analytic_signal_and_the_tilt_is_vertical_over_it():
    d = derivatives(dipole(), DX).layers
    c = NODD // 2
    peak = np.unravel_index(np.nanargmax(d["mag_as"]), d["mag_as"].shape)
    assert abs(peak[0] - c) <= 1 and abs(peak[1] - c) <= 1  # the analytic signal is largest right over the source
    assert np.nanmax(d["mag_as"]) > 3 * np.nanmedian(d["mag_as"])
    assert d["mag_tilt"][c, c] > 85.0  # +90 degrees over the top of the source, where the horizontal gradient vanishes
    assert d["mag_vd1"][c, c] > 0  # positive over a positive anomaly (downward convention)
    assert d["mag_tilt"][c, c + 13] < d["mag_tilt"][c, c] and np.nanmin(d["mag_tilt"]) < 0  # it falls away and changes sign


def test_tilt_is_independent_of_the_amplitude_of_the_source():
    weak = derivatives(dipole(amplitude=1e8), DX).layers["mag_tilt"]
    strong = derivatives(dipole(amplitude=1e10), DX).layers["mag_tilt"]
    assert np.allclose(weak, strong, atol=1e-6)  # the point of the tilt: weak and strong sources look alike


def test_texture_is_zero_over_a_flat_field_and_high_over_a_disturbed_one():
    flat = derivatives(np.full((64, 64), 55_000.0), DX).layers["mag_texture"]
    assert np.nanmax(flat) == pytest.approx(0.0, abs=1e-6)
    noisy = np.random.default_rng(0).normal(55_000, 40, (64, 64))
    assert np.nanmedian(derivatives(noisy, DX).layers["mag_texture"]) > 20


# --- nodata and bad grids ----------------------------------------------------------------------------------------


def test_nodata_stays_nodata_and_does_not_poison_its_neighbours():
    field = dipole()
    holed = field.copy()
    holed[10:30, 10:30] = np.nan
    base, with_hole = derivatives(field, DX).layers, derivatives(holed, DX).layers
    for name in NAMES:
        assert np.isnan(with_hole[name][10:30, 10:30]).all(), name  # restored to nodata
        assert np.isfinite(with_hole[name][60:70, 60:70]).all(), name  # no NaN spreads across the grid
    c = NODD // 2
    assert np.allclose(with_hole["mag_as"][c, c], base["mag_as"][c, c], rtol=0.05)  # far from the hole, same answer


def test_grids_that_cannot_be_differentiated_say_why():
    with pytest.raises(MagneticsError, match="too small"):
        derivatives(np.ones((5, 40)), DX)
    with pytest.raises(MagneticsError, match="no data"):
        derivatives(np.full((32, 32), np.nan), DX)


def test_all_named_layers_are_returned_on_the_input_grid():
    field = dipole()
    out = derivatives(field, DX)
    assert tuple(out.layers) == NAMES and out.dx == DX
    assert all(a.shape == field.shape for a in out.layers.values())
    assert np.array_equal(out.layers["mag_field"], field)  # the field itself is passed through unchanged
