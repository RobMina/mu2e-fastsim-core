"""Unit contracts and opt-in checks against a user-built CRY image."""
import ctypes as ct
import os

import numpy as np
import pytest

from mu2efastsim import cry_native as cry


class FakeLibrary:
    def __init__(self):
        self.destroyed = 0
        self.failed = False
        self.rows = (cry._Particle * 2)()
        self.rows[0] = cry._Particle(0, 13, 0, 100, 1, 2, 0, 0, 0, -1, 0.25)
        self.rows[1] = cry._Particle(0, -13, 0, 200, 3, 4, 0, 0, 0, -1, 0.25)
        self.times = (ct.c_double * 2)(0.25, 0.5)

    def m2cry_create(self, *_):
        return None if self.failed else 123

    def m2cry_generate(self, handle, count, rows, size, times, before):
        ct.cast(rows, ct.POINTER(ct.POINTER(cry._Particle)))[0] = self.rows
        ct.cast(size, ct.POINTER(ct.c_uint64))[0] = 2 if count else 0
        ct.cast(times, ct.POINTER(ct.POINTER(ct.c_double)))[0] = self.times
        ct.cast(before, ct.POINTER(ct.c_double))[0] = 0.0
        return -1 if self.failed else 0

    def m2cry_last_error(self):
        return b"backend error"

    def m2cry_destroy(self, handle):
        self.destroyed += 1


@pytest.fixture
def fake(tmp_path, monkeypatch):
    for h in (0, 2100, 11300):
        (tmp_path / f"cosmics_{h}.data").write_text("test fixture")
    lib = FakeLibrary()
    monkeypatch.setattr(cry, "_library", lambda _: lib)
    return tmp_path, lib


def test_particle_abi():
    assert ct.sizeof(cry._Particle) == 80
    assert cry._Particle.ke.offset == 16
    assert cry._Particle.t.offset == 72


def test_batch_copy_and_context(fake):
    path, lib = fake
    with cry.CrySource(data=path) as source:
        batch = source.next_showers(2)
        assert batch.particles["pdg"].tolist() == [13, -13]
        assert batch.live_seconds == 0.5
        lib.rows[0].pdg = 22
        lib.times[1] = 1.0
        assert batch.particles["pdg"][0] == 13
        assert batch.cumulative_seconds[-1] == 0.5
    assert lib.destroyed == 1
    source.close()
    assert lib.destroyed == 1
    with pytest.raises(RuntimeError, match="closed"):
        source.next_showers(1)


def test_zero_count(fake):
    path, _ = fake
    with cry.CrySource(data=path) as source:
        batch = source.next_showers(0)
    assert len(batch.particles) == len(batch.cumulative_seconds) == 0
    assert batch.live_seconds == 0


@pytest.mark.parametrize("kwargs", [
    {"seed": -1}, {"seed": 2**64}, {"seed": True}, {"subbox_m": 0},
    {"subbox_m": np.nan}, {"latitude_deg": 91}, {"altitude_m": 1},
    {"when": "2021-02-29"}, {"when": "1800-01-01"},
])
def test_invalid_settings(fake, kwargs):
    with pytest.raises(ValueError):
        cry.CrySource(data=fake[0], **kwargs)


@pytest.mark.parametrize("count", [-1, 100001, 1.5, True])
def test_invalid_count(fake, count):
    with cry.CrySource(data=fake[0]) as source:
        with pytest.raises(ValueError):
            source.next_showers(count)


def test_missing_tables(tmp_path):
    with pytest.raises(FileNotFoundError, match="cosmics_0"):
        cry.CrySource(data=tmp_path)


def test_backend_errors(fake):
    path, lib = fake
    with cry.CrySource(data=path) as source:
        lib.failed = True
        with pytest.raises(RuntimeError, match="backend error"):
            source.next_showers(1)
    with pytest.raises(RuntimeError, match="backend error"):
        cry.CrySource(data=path)


@pytest.mark.skipif(os.environ.get("MU2E_CRY_TEST_NATIVE") != "1", reason="requires user-built CRY shim")
def test_native_repeatability_and_batching():
    with cry.CrySource(seed=1234) as a, cry.CrySource(seed=1234) as b:
        all_rows = a.next_showers(20)
        first, second = b.next_showers(10), b.next_showers(10)
        merged = second.particles.copy()
        merged["shower"] += 10
        np.testing.assert_array_equal(all_rows.particles, np.concatenate([first.particles, merged]))
        np.testing.assert_array_equal(all_rows.cumulative_seconds,
                                      np.concatenate([first.cumulative_seconds, second.cumulative_seconds]))
        assert second.before_seconds == first.cumulative_seconds[-1]
        assert np.all(np.diff(all_rows.cumulative_seconds) >= 0)
        assert all_rows.live_seconds > 0
        assert np.all(np.isfinite(all_rows.particles["ke"]))
        assert np.all(all_rows.particles["ke"] >= 0)
        rows = all_rows.particles
        np.testing.assert_allclose(rows["u"]**2 + rows["v"]**2 + rows["w"]**2, 1, atol=1e-10)
        assert set(np.abs(rows["pdg"])) <= {11, 13, 22, 211, 321, 2112, 2212}
        with cry.CrySource(seed=5678) as c:
            different = c.next_showers(20)
        assert not np.array_equal(all_rows.particles, different.particles)
