"""Unit tests: helix stepping in a magnetic field (mu2efastsim.ehelix) and the slab stepper with a field."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mu2efastsim import ebrem as B                                       # noqa: E402
from mu2efastsim import ehelix as H                                      # noqa: E402
from mu2efastsim import eioni as I                                       # noqa: E402
from mu2efastsim import eslab as S                                       # noqa: E402
from mu2efastsim import materials as M                                   # noqa: E402


def _tracks(n, rng):
    u = rng.normal(size=(n, 3))
    return rng.normal(size=(n, 3)) * 10, u / np.linalg.norm(u, axis=1, keepdims=True)


def test_helix_matches_the_analytic_circle_and_pitch():
    # Bz = 1 T, e- (q = -1) of p = 100 MeV/c starting along x with a z component: circle radius p_T / (K B)
    p, Bz, s = 100.0, 1.0, np.array([0.0, 10.0, 250.0, 2000.0])
    u0 = np.array([0.8, 0.0, 0.6])
    x, u = H.helix(np.zeros((4, 3)), np.tile(u0, (4, 1)), p, -1.0, np.tile([0, 0, Bz], (4, 1)), s)
    R = p * 0.8 / (H.K * Bz)
    phi = s * 0.8 / R                       # e- in +z field turns counter-clockwise seen from +z
    assert x[:, 0] == pytest.approx(R * np.sin(phi), abs=1e-9)
    assert x[:, 1] == pytest.approx(R * (1 - np.cos(phi)), abs=1e-9)
    assert x[:, 2] == pytest.approx(0.6 * s, abs=1e-12)
    assert u[:, 2] == pytest.approx(0.6) and np.linalg.norm(u, axis=1) == pytest.approx(1.0)


def test_helix_steps_compose_and_zero_field_is_straight():
    rng = np.random.default_rng(1)
    x0, u0 = _tracks(50, rng)
    Bv = np.tile([0.3, -0.5, 1.0], (50, 1))
    xa, ua = H.helix(x0, u0, 50.0, 1.0, Bv, 300.0)
    xb, ub = x0, u0
    for _ in range(7):
        xb, ub = H.helix(xb, ub, 50.0, 1.0, Bv, 300.0 / 7)
    assert xb == pytest.approx(xa, abs=1e-9) and ub == pytest.approx(ua, abs=1e-12)
    xs, us = H.helix(x0, u0, 50.0, 1.0, np.zeros((50, 3)), 3.0)
    assert xs == pytest.approx(x0 + 3.0 * u0) and us == pytest.approx(u0)


def test_rk4_converges_to_the_helix_in_a_uniform_field():
    rng = np.random.default_rng(2)
    x0, u0 = _tracks(20, rng)
    Bv = np.array([0.0, 0.2, 1.0])
    xh, uh = H.helix(x0, u0, 30.0, -1.0, np.tile(Bv, (20, 1)), 100.0)
    errs = []
    for nstep in (10, 20):
        x, u = x0, u0
        for _ in range(nstep):
            x, u = H.rk4(x, u, 30.0, -1.0, lambda xx: np.tile(Bv, (xx.shape[0], 1)), 100.0 / nstep)
        errs.append(np.abs(x - xh).max())
    assert errs[1] < 1e-4 and errs[0] / errs[1] > 12        # fourth order


def test_to_plane_finds_the_first_crossing():
    # B along x: a track along +z curls in the y-z plane, radius R; plane z = 0.5 R is reached at s = R asin(0.5)
    n, p, Bx = 4, 20.0, 1.5
    R = p / (H.K * Bx)
    x0 = np.zeros((n, 3))
    u0 = np.tile([0.0, 0.0, 1.0], (n, 1))
    Bv = np.tile([Bx, 0.0, 0.0], (n, 1))
    s = H.to_plane(x0, u0, p, -1.0, Bv, np.array([0.5 * R, 0.5 * R, 2 * R, -0.1]), np.array([10 * R, 0.1, 10 * R,
                                                                                                10 * R]))
    assert s[0] == pytest.approx(R * np.arcsin(0.5), rel=1e-9)
    assert np.isinf(s[1]) and np.isinf(s[2])               # beyond s_max; never reached (max z = R)
    assert s[3] == pytest.approx(R * (np.pi + np.arcsin(0.1 / R)), rel=1e-9)   # comes back down through z < 0
    xe, _ = H.helix(x0[:1], u0[:1], p, -1.0, Bv[:1], s[:1])
    assert xe[0, 2] == pytest.approx(0.5 * R, abs=1e-9)


def test_slab_in_a_transverse_field_bends_and_loses_along_the_curved_path():
    # 10 MeV e- through 20 mm of air in By = 1 T: path R asin(L / R), continuous loss dE/dx (restricted) x path
    n, T0, L, By, cut = 4000, 10.0, 20.0, 1.0, 0.01
    air = M.BY_NAME["G4_AIR"]
    r = S.electron_slab(np.random.default_rng(3), air, np.full(n, T0), np.tile([0.0, 0.0, 1.0], (n, 1)), L, cut, cut,
                        field=[0.0, By, 0.0])
    p = np.sqrt(T0 * (T0 + 2 * B.ME))
    R = p / (H.K * By)
    s = R * np.arcsin(L / R)
    assert r.path.mean() == pytest.approx(s, rel=3e-3)       # radius shrinks by the ~0.5 % energy lost
    dedx = (I.dedx_restricted(air, T0, cut) + B.dedx_restricted(air, T0, cut))[0]
    assert r.continuous.mean() == pytest.approx(dedx * r.path.mean(), rel=0.02)
    straight = S.electron_slab(np.random.default_rng(3), air, np.full(n, T0), np.tile([0.0, 0.0, 1.0], (n, 1)), L,
                               cut, cut)
    per_mm = r.continuous.mean() / r.path.mean()
    assert per_mm == pytest.approx(straight.continuous.mean() / straight.path.mean(), rel=0.02)
    x = r.position[:, 0]
    assert np.median(x) == pytest.approx(R * (1 - np.cos(s / R)), rel=0.01)    # F = -e (z x y): e- bends to +x
    assert np.allclose(r.position[:, 2], L, atol=1e-9) and not r.reflected.any()


def test_low_momentum_track_curls_back_out_of_the_entry_face():
    # 1 MeV e- in 0.5 T vacuum-like gas entering at 60 degrees to the field: it loops and leaves through z = 0
    n = 200
    air = M.BY_NAME["G4_AIR"]
    d = np.tile([np.sin(np.pi / 3), 0.0, np.cos(np.pi / 3)], (n, 1))
    r = S.electron_slab(np.random.default_rng(4), air, np.full(n, 1.0), d, 100.0, 1e3, 1e3,
                        field=[0.0, 0.5, 0.0])
    assert r.reflected.all() and np.allclose(r.position[:, 2], 0.0, atol=1e-9)


def test_grid_field_is_exact_for_a_linear_field_and_zero_outside():
    lo, d, n = np.array([-100.0, -50.0, 0.0]), np.array([10.0, 10.0, 20.0]), np.array([21, 11, 11])
    g = np.stack(np.meshgrid(*(lo[k] + d[k] * np.arange(n[k]) for k in range(3)), indexing="ij"), axis=-1)
    Bg = np.stack([0.01 * g[..., 1], -0.002 * g[..., 0], 1.0 + 0.001 * g[..., 2]], axis=-1)
    f = H.grid_field(lo, d, n, Bg)
    x = np.array([[3.3, -7.1, 55.5], [-99.0, 49.0, 199.0], [500.0, 0.0, 0.0]])
    b = f(x)
    assert b[:2] == pytest.approx(np.stack([0.01 * x[:2, 1], -0.002 * x[:2, 0], 1.0 + 0.001 * x[:2, 2]], axis=1))
    assert (b[2] == 0).all()


def test_propagate_to_a_plane_conserves_and_agrees_between_methods():
    # 105 MeV e- in a solenoid-like field with a gradient: helix steps and RK4 reach the plane at the same point
    rng = np.random.default_rng(5)
    n = 200
    ct = rng.uniform(0.5, 0.8, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    u0 = np.stack([np.sqrt(1 - ct ** 2) * np.cos(phi), np.sqrt(1 - ct ** 2) * np.sin(phi), ct], axis=1)
    field = lambda x: np.stack([-0.5e-4 * x[:, 0], -0.5e-4 * x[:, 1], 1.0 + 1e-4 * x[:, 2]], axis=1)  # div B = 0
    p = 105.5
    zp = 500.0   # several turns already; the step loop is per-step bound, so a farther plane only costs time
    xh, uh, sh, rh = H.propagate(np.zeros((n, 3)), u0, p, -1.0, field, 2.0, zp, 1e5)
    xr, ur, sr, rr = H.propagate(np.zeros((n, 3)), u0, p, -1.0, field, 2.0, zp, 1e5, method="rk4")
    assert rh.all() and rr.all() and np.allclose(xh[:, 2], zp) and np.allclose(xr[:, 2], zp)
    assert np.abs(xh - xr).max() < 0.05 and np.abs(sh - sr).max() < 0.05
