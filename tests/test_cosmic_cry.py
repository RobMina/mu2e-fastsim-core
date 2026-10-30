"""Stock wrapper geometry, cuts, rejected-shower carry and event-clock contracts."""
import numpy as np
import pytest

from mu2efastsim.cosmic_cry import CosmicConfig, CosmicSource, highest_intersection, project
from mu2efastsim.cry_native import CryBatch, _Particle


def cfg():
    return CosmicConfig((0, 20, 0), (-10, -10, -10), (10, 10, 10),
                        (-100, -15, -100), (100, 15, 100), "fixture", max_shower_mev=1000)


def batch(ke=(100, 200), showers=(0, 1), pdg=(13, -13), clock=(1, 2)):
    out = np.zeros(len(ke), dtype=np.dtype(_Particle))
    out["ke"], out["shower"], out["pdg"] = ke, showers, pdg
    out["w"] = -1
    out["t"] = np.asarray(clock)[out["shower"]]
    return CryBatch(out, np.asarray(clock, float), 0)


def test_five_faces_lines_and_parallel():
    pos = np.array([[0, 20, 0], [0, -20, 0], [20, 0, 0], [20, -20, 0], [0, 0, 0]])
    direction = np.array([[0, -1, 0], [0, -1, 0], [-1, 0, 0], [0, 1, 0], [0, 0, 0]])
    hit, mask = highest_intersection(pos, direction, (-10, -10, -10), (10, 10, 10))
    assert mask.tolist() == [True, True, True, False, False]
    np.testing.assert_array_equal(hit[:3], [[0, 10, 0], [0, 10, 0], [-10, 0, 0]])


def test_map_cuts_total_energy_and_world_projection():
    b = batch((49, 50, 100), (0, 0, 1), (13, 22, -13))
    rows, sums = project(b, cfg())
    assert rows.pdg.tolist() == [22, -13]
    np.testing.assert_allclose(sums, [50, 205.658367])
    np.testing.assert_allclose(rows.y, 15)
    np.testing.assert_allclose(rows.py, -np.sqrt(rows.e**2 - np.array([0, 105.658367])**2))
    np.testing.assert_array_equal(rows.t, [0, 0])


class Source:
    def __init__(self, batches):
        self.batches = iter(batches)
    def next_showers(self, count):
        return next(self.batches)
    def close(self):
        pass


def test_overenergy_carry_through_empty_terminal_and_clock():
    b1 = batch((1200,), (0,), (13,), (1,))
    b2 = batch((100,), (1,), (22,), (2, 3))
    with CosmicSource(cfg(), raw_batch=1, source=Source([b1, b2])) as source:
        gen, sr, receipt = source.generate(2)
    assert gen.event.tolist() == [1, 2]
    assert gen.pdg.tolist() == [13, 22]
    assert sr.livetime.iloc[0] == 3
    assert receipt["overenergy_raw_showers_drawn"] == 1
    assert receipt["accumulated_particle_rows"] == 1


def test_shared_event_time_and_buffering():
    b = batch((100, 200, 100), (0, 0, 1), (13, -13, 22))
    with CosmicSource(cfg(), source=Source([b])) as source:
        first, sr1, r1 = source.generate(1)
        second, sr2, r2 = source.generate(1, first_event=2)
    assert r1["event_range"] == {"run": 1400, "subrun": 1, "first_event": 1, "last_event": 1}
    assert r2["event_range"] == {"run": 1400, "subrun": 1, "first_event": 2, "last_event": 2}
    assert first.idx.tolist() == [0, 1]
    assert first.t.iloc[0] == first.t.iloc[1]
    assert second.event.tolist() == [2]
    assert sr1.livetime.iloc[0] == sr2.livetime.iloc[0] == 1


def test_accepted_events_invariant_to_raw_batch_boundaries():
    b = batch((1200, 100, 200, 100), (0, 1, 1, 2), (13, -13, 22, 22), (1, 2, 3))
    left = batch((1200,), (0,), (13,), (1,))
    right = batch((100, 200, 100), (0, 0, 1), (-13, 22, 22), (2, 3))
    with CosmicSource(cfg(), source=Source([b])) as a, CosmicSource(cfg(), source=Source([left, right])) as s:
        ga, sa, _ = a.generate(2)
        gs, ss, _ = s.generate(2)
    import pandas as pd
    pd.testing.assert_frame_equal(ga, gs)
    pd.testing.assert_frame_equal(sa, ss)


def test_extracted_factory_rejects_unknown_gdml(tmp_path):
    path = tmp_path / "different.gdml"
    path.write_text("<gdml/>")
    with pytest.raises(ValueError, match="signature"):
        CosmicConfig.extracted(path)


def test_bad_config():
    with pytest.raises(ValueError):
        CosmicConfig((0, 0, 0), (1, 1, 1), (0, 0, 0), (-1, -1, -1), (1, 1, 1), "bad")
