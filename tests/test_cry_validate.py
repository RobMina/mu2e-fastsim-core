"""Exposure and target-position summaries use every generated primary and accepted event."""
import numpy as np
import pandas as pd

from mu2efastsim.cosmic_cry import CosmicConfig
from mu2efastsim.cry_validate import _summarise


def test_summary_counts_and_target():
    config = CosmicConfig((0, 20, 0), (-10, -10, -10), (10, 10, 10),
                          (-100, -15, -100), (100, 15, 100), "fixture")
    gen = pd.DataFrame({"run": [1, 1, 1], "subrun": [1, 1, 1], "event": [1, 1, 2],
                        "idx": [0, 1, 0], "pdg": [22, 22, 22], "e": [100., 100., 100.],
                        "x": [0., 1., 0.], "y": [15., 15., 15.], "z": [0., 1., 0.],
                        "px": [0., 0., 0.], "py": [-100., -100., -100.], "pz": [0., 0., 0.],
                        "t": [5., 5., 10.]})
    summary = _summarise(gen, 2, config)
    assert summary["metrics"]["accepted_per_s"] == 1
    assert summary["metrics"]["rate_pdg_22"] == 1.5
    assert summary["metrics"]["multiprimary_fraction"] == .5
    assert summary["metrics"]["multimuon_fraction"] == 0
    assert summary["metrics"]["multi_dt_max_ns"] == 0
    assert np.isclose(sum(summary["histograms"]["22:target_y"]["counts_per_event"]), 1.5)

    expected = np.sqrt(2)/1000
    assert summary["metrics"]["multi_pair_world_xz_median_m"] == expected
    assert summary["metrics"]["multi_pair_target_xz_p90_m"] == expected
    assert sum(summary["histograms"]["multi_pair_world_xz"]["counts_per_event"]) == .5
