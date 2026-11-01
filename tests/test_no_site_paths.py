"""No user name and no site, user or storage path in this repository (shared checker: mu2efastsim.testing)."""
from pathlib import Path

from mu2efastsim.testing import check_site_paths, site_path_self_test

ROOT = Path(__file__).resolve().parents[1]


def test_patterns_catch_what_they_should():
    site_path_self_test()


def test_no_user_names_or_site_paths_in_tracked_files():
    bad = check_site_paths(ROOT)
    assert not bad, "site paths or user names in the repository:\n" + "\n".join(bad)
