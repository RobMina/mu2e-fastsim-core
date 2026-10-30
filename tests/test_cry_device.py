"""Device CRY table/model/kernel contracts; real GPU/stock comparisons run separately."""
import os
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from mu2efastsim import cry_device as D, cry_tables as T
from mu2efastsim.cosmic_cry import CosmicConfig, highest_intersection


TABLE = Path(os.environ.get("CRYDATAPATH", "/opt/cry/data")) / "cosmics_0.data"


@pytest.fixture
def setup():
    if not TABLE.exists():
        pytest.skip("verified CRY table not installed")
    return D.device_tables(TABLE)


def test_strict_parser_pdf_mean():
    data = T.parse("% comment\npdf a::b[1,3,dis] = {{1 2 1} {0 0 0}}; binning b={1 2 3};")
    np.testing.assert_allclose(data["a"].mean(), [2, 0])
    assert data["a"].weights.shape == (2, 3)
    with pytest.raises(ValueError):
        data["a"].cdf()
    for text in ("pdf a::b[0,1,lin] = {-1 1};", "parameter a={nan};", "unknown a={1};",
                 "parameter a={1}; parameter a={2};"):
        with pytest.raises(ValueError):
            T.parse(text)


def test_table_signature_fail_closed(tmp_path):
    path = tmp_path / "table"
    path.write_text("pdf a::b[0,1,lin]={1};")
    with pytest.raises(ValueError, match="signature"):
        T.load(path)


def test_primary_rates_and_distribution_dimensions(setup):
    tables, metadata = setup
    assert metadata["lifetime_s"] == pytest.approx(5.0136379137012805e-8, rel=1e-12)
    assert tables["primary"].cdf.shape == (1, 10000)
    assert tables["multiplicity"].cdf.shape == (33, 1000)
    assert tables["energy"].cdf.shape == (7*33, 85)
    assert tables["time"].cdf.shape == (7*85, 45)


def test_pdf_inverse_and_within_bin_semantics():
    pdf = T.Pdf("", 1, 100, "log", np.array([[1., 2., 0., 1.]]))
    p = D._pdf(pdf)
    u = jnp.array([0., .25, .5, .75, .9999])
    row = jnp.zeros(5, jnp.int32)
    result, valid = jax.jit(D.draw)(p, row, u, jnp.full(5, .5))
    np.testing.assert_array_equal(D.inverse(p.cdf, row, u), [0, 1, 1, 3, 3])
    np.testing.assert_allclose(result, 10**((np.array([0,1,1,3,3])+.5)/2))
    assert np.all(valid)


def test_device_projection_matches_stock_geometry():
    position = np.array([[0.,20,0], [0,-20,0], [20,0,0], [20,-20,0], [0,0,0]])
    direction = np.array([[0.,-1,0], [0,-1,0], [-1,0,0], [0,1,0], [0,0,0]])
    ref, mask = highest_intersection(position, direction, (-10,-10,-10), (10,10,10))
    got, valid = jax.jit(D.intersection)(jnp.array(position), jnp.array(direction), jnp.array([-10.,-10,-10]), jnp.array([10.,10,10]))
    np.testing.assert_array_equal(valid, mask)
    np.testing.assert_array_equal(np.asarray(got)[mask], ref[mask])


def test_raw_device_bounded_pool_repeatability_and_units(setup):
    tables, _ = setup
    key = jax.random.key(144)
    a = D.raw(tables, key, 0., showers=4096, capacity=20000)
    b = D.raw(tables, key, 0., showers=4096, capacity=20000)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])
    assert not a["overflow"] and not a["invalid_pdf"]
    mask = np.asarray(a["mask"])
    assert int(a["particles"]) == int(np.sum(a["multiplicity"])) == np.sum(mask)
    np.testing.assert_allclose(np.sum(np.asarray(a["direction"])[mask]**2, axis=1), 1, atol=1e-14)
    assert np.all(np.abs(np.asarray(a["position"])[mask, :2]) <= 150)
    assert np.all(np.diff(a["clocks"]) > 0)
    assert set(np.abs(np.asarray(a["pdg"])[mask])) <= set(T.PDG)
    small = D.raw(tables, key, 0., showers=4096, capacity=1)
    assert small["overflow"]
    assert int(small["particles"]) == int(a["particles"])


def test_device_source_admission():
    config = CosmicConfig((0,20000,0), (-100,-5000,-100), (100,10000,100),
                          (-200000,-10000,-200000), (200000,15000,200000), "fixture")
    if jax.default_backend() != "gpu":
        with pytest.raises(ValueError, match="GPU"):
            D.DeviceCosmicSource(config)
    for seed in (-1, 2**64, True):
        with pytest.raises(ValueError):
            D.DeviceCosmicSource(config, seed=seed, require_gpu=False)


def test_accepted_pool_overflow_and_host_boundary(setup):
    config = CosmicConfig((0,20000,0), (-10000,-5000,-10000), (10000,10000,10000),
                          (-200000,-10000,-200000), (200000,15000,200000), "fixture")
    with D.DeviceCosmicSource(config, seed=18, raw_batch=10000, particle_capacity=50000,
                              selected_capacity=512, table=TABLE, require_gpu=False) as source:
        gen, subruns, receipt = source.generate(20)
    assert gen.event.nunique() == 20 and receipt["generator"] == "jax-cry-1.7"
    assert receipt["device_kernel_wall_s"] > 0
    assert receipt["sampling"]["table_sha256"] == T.SEA_LEVEL_SHA256
    assert subruns.livetime.iloc[0] == receipt["live_seconds"] > 0
    with D.DeviceCosmicSource(config, seed=18, raw_batch=1000, particle_capacity=1,
                              table=TABLE, require_gpu=False) as source:
        with pytest.raises(RuntimeError, match="overflow"):
            source.generate(1)
        with pytest.raises(RuntimeError, match="failed"):
            source.generate(1)


def test_device_overenergy_carry_across_empty_terminal_and_batches():
    config = CosmicConfig((0,20,0), (-10,-10,-10), (10,10,10), (-100,-15,-100), (100,15,100),
                          "fixture", max_shower_mev=1000, offset_ns=(0, 1e-300))
    def batch(energies, shower, clocks):
        n = len(energies)
        return {"pdg": jnp.full(n,13), "ke": jnp.array(energies, float), "shower": jnp.array(shower),
                "mask": jnp.ones(n,bool), "position": jnp.zeros((n,3)),
                "direction": jnp.tile(jnp.array([0.,0.,-1.]), (n,1)),
                "t": jnp.array(clocks)[jnp.array(shower)], "clocks": jnp.array(clocks),
                "overflow": jnp.array(False), "invalid_pdf": jnp.array(False), "particles": jnp.array(n)}
    cfg = D._config(config)
    first = D.select(batch([1200.], [0], [1.]), cfg, jax.random.key(1), jnp.zeros((8,9)), jnp.array(0),
                     showers=1, selected_capacity=8)
    assert first["events"] == first["row_count"] == 0 and first["pending_count"] == 1
    second = D.select(batch([100.], [1], [2.,3.]), cfg, jax.random.key(2), first["pending"], first["pending_count"],
                      showers=2, selected_capacity=8)
    assert second["events"] == second["row_count"] == 2 and second["pending_count"] == 0
    np.testing.assert_allclose(np.asarray(second["rows"])[:2,1], [1305.658367, 205.658367])
    np.testing.assert_array_equal(np.asarray(second["slots"])[:2], [0,1])
    np.testing.assert_array_equal(np.asarray(second["clocks"])[:2], [2.,3.])


@pytest.mark.parametrize("species", range(7))
def test_conditional_energy_sampler_shared_primary_and_exact_pdf_pmf(setup, species):
    tables, _ = setup
    # Fixed primary bin tests the conditional PDF without marginal cancellation.
    primary_bin = 15
    pdf = T.load(TABLE)["tables"][T.SPECIES[species]+"KEDist"]
    row = jnp.full(20000, species*33+primary_bin, jnp.int32)
    energy, valid = D.draw(tables["energy"], row, D.uniform(jax.random.key(species), 0, (20000,)),
                           D.uniform(jax.random.key(species), 1, (20000,)), jnp.full(20000, species))
    observed = np.histogram(np.log10(energy), np.linspace(-9, 8, 86))[0]
    probability = pdf.weights[primary_bin]/pdf.weights[primary_bin].sum()
    variance = 20000*probability*(1-probability)
    np.testing.assert_array_less(np.abs(observed-20000*probability), 7*np.sqrt(variance)+4)
    assert np.all(valid)


def test_device_program_has_no_host_sampling_callback(setup, monkeypatch):
    from mu2efastsim import cosmic_cry
    tables, _ = setup
    monkeypatch.setattr(cosmic_cry.CosmicSource, "_fill", lambda _: pytest.fail("host CRY generation reached"))
    config = CosmicConfig((0,20000,0), (-10000,-5000,-10000), (10000,10000,10000),
                          (-200000,-10000,-200000), (200000,15000,200000), "fixture")
    jaxpr = str(jax.make_jaxpr(lambda key: D.accepted(tables, D._config(config), key, 0.,
                        jnp.zeros((64,9)), jnp.array(0), showers=64, capacity=512, selected_capacity=64))(jax.random.key(10)))
    for forbidden in ("pure_callback", "io_callback", "host_callback"):
        assert forbidden not in jaxpr
    with D.DeviceCosmicSource(config, seed=16, raw_batch=10000, selected_capacity=512,
                              table=TABLE, require_gpu=False) as source:
        gen, _, receipt = source.generate(2)
        assert len(gen) >= 2
        if jax.default_backend() == "gpu":
            assert source.device.platform == "gpu"
            assert receipt["device"].startswith("cuda")
    with pytest.raises(RuntimeError, match="closed"):
        source.generate(1)



def test_full_uint64_seed_bits_change_device_stream(setup):
    config = CosmicConfig((0,20000,0), (-10000,-5000,-10000), (10000,10000,10000),
                          (-200000,-10000,-200000), (200000,15000,200000), "fixture")
    with D.DeviceCosmicSource(config, seed=5, table=TABLE, require_gpu=False) as low:
        key_low = jax.random.key_data(low.key)
    with D.DeviceCosmicSource(config, seed=5+2**32, table=TABLE, require_gpu=False) as high:
        key_high = jax.random.key_data(high.key)
    assert not np.array_equal(key_low, key_high)
