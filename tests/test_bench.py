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
