"""mu2efastsim.bench: pass timing, setup/warm-rate/N10 arithmetic, device sync, nvidia-smi parsing and the CLI."""
import json
import sys
import textwrap

import pytest

from mu2efastsim import bench


def fake_clock(pass_seconds):
    """A clock whose consecutive begin/end pairs differ by the given pass seconds."""
    ticks, t = [], 0.0
    for s in pass_seconds:
        ticks += [t, t + s]
        t += s + 1.0
    it = iter(ticks)
    return lambda: next(it)


def test_summary_arithmetic():
    calls = []
    r = bench.run(lambda n: calls.append(n), 1000, passes=3, args=(1000,), clock=fake_clock([12.0, 2.0, 2.0]),
                  label="test device", dtype="float32")
    assert calls == [1000, 1000, 1000]
    assert r["pass_s"] == [12.0, 2.0, 2.0]
    assert r["warm_rate"] == pytest.approx(500.0)
    assert r["setup_s"] == pytest.approx(10.0)
    assert r["n10"] == pytest.approx(9 * 10.0 * 500.0)
    assert r["device"] == "test device" and r["dtype"] == "float32" and r["dtype_source"] == "argument"


def test_setup_never_negative_and_validation():
    r = bench.summarise([1.0, 2.0, 2.0], 10)
    assert r["setup_s"] == 0.0 and r["n10"] == 0.0
    with pytest.raises(ValueError):
        bench.time_passes(lambda: None, 10, passes=1)
    with pytest.raises(ValueError):
        bench.time_passes(lambda: None, 0)


def test_sync_runs_before_the_clock_stops():
    order = []

    class Result:
        def block_until_ready(self):
            order.append("sync")
            return self

    def clock():
        order.append("clock")
        return float(len(order))

    bench.time_passes(Result, 5, passes=2, sync=bench.device_sync if "jax" not in sys.modules else
                      (lambda r: r.block_until_ready()), clock=clock)
    assert order == ["clock", "sync", "clock"] * 2


NVSMI_MIG = textwrap.dedent("""\
    GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition (UUID: GPU-aaaa)
      MIG 1g.24gb     Device  0: (UUID: MIG-1111)
      MIG 1g.24gb     Device  1: (UUID: MIG-2222)
    """)
NVSMI_TWO = textwrap.dedent("""\
    GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-0000)
    GPU 1: NVIDIA A100-SXM4-40GB (UUID: GPU-1111)
    """)


def test_parse_nvidia_smi():
    assert bench.parse_nvidia_smi(NVSMI_MIG, "MIG-2222") == "NVIDIA RTX PRO 6000 Blackwell Server Edition, MIG 1g.24gb"
    assert bench.parse_nvidia_smi(NVSMI_MIG, None) == "NVIDIA RTX PRO 6000 Blackwell Server Edition, MIG 1g.24gb"
    assert bench.parse_nvidia_smi(NVSMI_TWO, "1") == "NVIDIA A100-SXM4-40GB"
    assert bench.parse_nvidia_smi(NVSMI_TWO, "GPU-1111") == "NVIDIA A100-SXM4-40GB"
    assert bench.parse_nvidia_smi(NVSMI_TWO, None) == "NVIDIA A100-SXM4-80GB"
    assert bench.parse_nvidia_smi("", None) is None


def test_cpu_label_names_the_model():
    label = bench.device_label("cpu")
    assert "threads available" in label and len(label) > len(" ( threads available)")


def test_resolve():
    assert bench.resolve("json:dumps") is json.dumps
    for bad in ("json", ":dumps", "json:"):
        with pytest.raises(ValueError):
            bench.resolve(bad)


def test_cli_with_prepare_and_kwargs(tmp_path, monkeypatch, capsys):
    (tmp_path / "toyflow.py").write_text(textwrap.dedent("""\
        SEEN = []
        def load(n):
            return list(range(n))
        def step(events, scale=1):
            SEEN.append(len(events))
            return sum(events) * scale
        """))
    monkeypatch.syspath_prepend(str(tmp_path))
    out = tmp_path / "out" / "bench.json"
    rc = bench.main(["toyflow:step", "--events", "50", "--prepare", "toyflow:load", "--kwargs", '{"scale": 2}',
                     "--workflow", "toy", "--boundary", "list -> sum", "--dtype", "float64",
                     "--device-label", "unit test", "--out", str(out)])
    assert rc == 0
    import toyflow
    assert toyflow.SEEN == [50, 50, 50]
    r = json.loads(out.read_text())
    assert r["events"] == 50 and len(r["pass_s"]) == 3 and r["prepare_s"] >= 0
    assert r["workflow"] == "toy" and r["boundary"] == "list -> sum" and r["kwargs"] == {"scale": 2}
    assert r["device"] == "unit test" and r["n10"] == pytest.approx(9 * r["setup_s"] * r["warm_rate"])
    assert "warm" in capsys.readouterr().out


def test_cli_rejects_non_object_kwargs(tmp_path):
    with pytest.raises(SystemExit):
        bench.main(["json:dumps", "--events", "1", "--kwargs", "[1]"])


def test_fresh_inputs_draw_every_pass_outside_the_clock():
    drawn, seen = [], []

    def prepare(n, seed=None):
        drawn.append(seed)
        return [seed] * n

    r = bench.run(lambda x: seen.append(list(x)), 4, passes=3, clock=fake_clock([5.0, 1.0, 1.0]), label="t",
                  inputs=bench.fresh_inputs(prepare, 4, seed=7), seeds=[7, 8, 9])
    assert drawn == [7, 8, 9] and seen == [[7] * 4, [8] * 4, [9] * 4]
    assert r["pass_s"] == [5.0, 1.0, 1.0] and r["setup_s"] == pytest.approx(4.0)
    assert r["inputs"] == "fresh" and r["seeds"] == [7, 8, 9]
    assert len(r["prepare_pass_s"]) == 3 and r["prepare_s"] == pytest.approx(sum(r["prepare_pass_s"]))


def test_fixed_inputs_stay_the_default():
    r = bench.run(lambda n: n, 10, passes=2, args=(10,), clock=fake_clock([2.0, 1.0]), label="t")
    assert r["inputs"] == "fixed" and r["seeds"] is None and r["prepare_pass_s"] is None


def test_fresh_inputs_refuse_a_prepare_without_seed():
    assert bench.accepts_seed(lambda n, seed=1: n) and bench.accepts_seed(lambda n, **kw: n)
    assert not bench.accepts_seed(lambda n: n) and not bench.accepts_seed(lambda n, *, other=1: n)
    with pytest.raises(bench.FreshInputError, match="takes no seed"):
        bench.fresh_inputs(lambda n: list(range(n)), 3)


def test_fresh_inputs_refuse_a_cached_input():
    cached = [1, 2, 3]
    inputs = bench.fresh_inputs(lambda n, seed=None: cached, 3)
    with pytest.raises(bench.FreshInputError, match="same object"):
        bench.run(lambda x: x, 3, passes=3, label="t", inputs=inputs)


TOYFLOW = textwrap.dedent("""\
    SEEN = []
    def load(n):
        return list(range(n))
    def load_seeded(n, seed=0):
        return [seed] * n
    def step(events, scale=1):
        SEEN.append(list(events))
        return sum(events) * scale
    """)


def test_cli_fresh_inputs(tmp_path, monkeypatch, capsys):
    (tmp_path / "toyfresh.py").write_text(TOYFLOW)
    monkeypatch.syspath_prepend(str(tmp_path))
    out = tmp_path / "bench.json"
    rc = bench.main(["toyfresh:step", "--events", "3", "--prepare", "toyfresh:load_seeded", "--fresh-inputs",
                     "--seed", "4", "--device-label", "unit test", "--out", str(out)])
    assert rc == 0
    import toyfresh
    assert toyfresh.SEEN == [[4] * 3, [5] * 3, [6] * 3]
    r = json.loads(out.read_text())
    assert r["inputs"] == "fresh" and r["seeds"] == [4, 5, 6] and len(r["prepare_pass_s"]) == 3
    assert "fresh inputs (seeds 4, 5, 6)" in capsys.readouterr().out


def test_cli_fresh_inputs_fail_loudly(tmp_path, monkeypatch, capsys):
    (tmp_path / "toyrefuse.py").write_text(TOYFLOW)
    monkeypatch.syspath_prepend(str(tmp_path))
    out = tmp_path / "bench.json"
    rc = bench.main(["toyrefuse:step", "--events", "3", "--prepare", "toyrefuse:load", "--fresh-inputs",
                     "--device-label", "unit test", "--out", str(out)])
    import toyrefuse
    assert rc == 2 and toyrefuse.SEEN == [] and not out.exists()
    assert "toyrefuse:load takes no seed" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        bench.main(["toyrefuse:step", "--events", "3", "--fresh-inputs"])


def test_jax_compiles_counted_per_pass():
    """Fixed inputs recompile nothing after pass 1; new shapes every pass compile in every pass."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    def step(x):
        return jnp.cumsum(jnp.sin(x)) * 2.0

    fixed = bench.run(step, 8, passes=3, args=(jnp.ones(8),), label="t")
    assert fixed["compile_source"] == "jax.monitoring"
    assert [c["backend_compiles"] for c in fixed["compile_per_pass"][1:]] == [0, 0]
    assert fixed["compile_per_pass"][0]["backend_compiles"] >= 1

    fresh = bench.run(step, 8, passes=3, label="t", seeds=[1, 2, 3],
                      inputs=bench.fresh_inputs(lambda n, seed: jnp.ones(n + 10 * seed), 8))
    for counter in fresh["compile_per_pass"]:
        assert counter["backend_compiles"] >= 1 and counter["backend_compile_s"] > 0
        assert counter["traces"] >= 1
    assert "JAX compiles per pass" in bench.format_report(fresh)
