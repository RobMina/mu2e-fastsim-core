"""Shared pytest hooks for the fastsim repositories (core, crv, tracker, umbrella).

Each repository's ``tests/conftest.py`` imports these hooks, so one test process stays bounded in memory:

- After every test module, JAX's compilation caches are cleared and garbage is collected. A suite otherwise keeps
  every compiled XLA program and its buffers for the rest of the process, which is how a full run grows to several GB.
- ``@pytest.mark.slow`` marks statistical validations with large samples (GB-scale or tens of seconds), JAX
  duplicates of a contract another per-PR test already checks, and campaign checks. They are skipped unless
  ``--runslow`` is given (or selected explicitly with ``-m slow``); the scheduled CI run passes ``--runslow``.
- ``@pytest.mark.pr_params("id", ...)`` on a parametrized test keeps only the listed parameter ids per PR; the other
  ids are marked slow, so every case still runs with ``--runslow``.
- ``@pytest.mark.weekly`` marks validations too long for the nightly run (tens of minutes on a CI runner). They are
  skipped unless ``--runweekly`` is given (or selected with ``-m weekly``); ``--runslow`` does not select them. The
  weekly scheduled CI run passes ``--runweekly -m weekly``.

Tiers: per PR (no option), nightly (``--runslow``: per-PR + slow), weekly (``--runweekly -m weekly``).
- ``XLA_PYTHON_CLIENT_PREALLOCATE`` defaults to false, so a test run on a GPU host does not reserve most of the card.
- ``MU2EFS_DEVICE_FLOAT`` defaults to float64: the suites check exact device-versus-NumPy parity, which holds in float64
  only. The float32 production default has its own tests (float32 subprocesses); ``MU2EFS_DEVICE_FLOAT=float32`` runs a
  whole suite in float32 mode.

Selecting the tests a change affects is ``python -m mu2efastsim.testselect`` (see that module).

``check_site_paths`` is the site-path and user-name scan every repository's ``tests/test_no_site_paths.py`` runs.
"""
import fnmatch
import gc
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MU2EFS_DEVICE_FLOAT", "float64")
if "mu2efastsim.precision" in sys.modules and sys.modules["mu2efastsim.precision"].NAME != os.environ["MU2EFS_DEVICE_FLOAT"]:
    raise RuntimeError("mu2efastsim.precision was imported before the test hooks; import mu2efastsim.testing first")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="also run @pytest.mark.slow statistical validations (large samples, GB-scale memory)")
    parser.addoption("--runweekly", action="store_true", default=False,
                     help="also run @pytest.mark.weekly validations (tens of minutes; not selected by --runslow)")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: large-sample statistical validation; skipped unless --runslow")
    config.addinivalue_line("markers", "pr_params(*ids): parameter ids kept per PR; the other ids are slow")
    config.addinivalue_line("markers", "weekly: validation too long for the nightly run; skipped unless --runweekly")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    for item in items:
        keep = item.get_closest_marker("pr_params")
        callspec = getattr(item, "callspec", None)
        if keep is not None and callspec is not None and callspec.id not in keep.args:
            item.add_marker(pytest.mark.slow)
    selected = config.getoption("-m") or ""
    runslow = config.getoption("--runslow") or "slow" in selected
    runweekly = config.getoption("--runweekly") or "weekly" in selected
    skip_slow = pytest.mark.skip(reason="slow validation: run with --runslow")
    skip_weekly = pytest.mark.skip(reason="weekly validation: run with --runweekly")
    for item in items:
        if "weekly" in item.keywords:
            if not runweekly:
                item.add_marker(skip_weekly)
        elif "slow" in item.keywords and not runslow:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True, scope="module")
def _release_jax_caches():
    """Drop compiled programs and their buffers when a test module finishes."""
    yield
    jax = sys.modules.get("jax")
    if jax is not None:
        jax.clear_caches()
    gc.collect()


# ---------------------------------------------------------------------------------------------------------------------
# Site-path checker shared by every repository's tests/test_no_site_paths.py.
#
# No user name and no site, user or storage path may be written into a repository: site paths come from configuration
# (mu2efastsim.site, the umbrella's site.env.example). Allowed: /cvmfs, system paths (/usr, /bin, /dev, /proc, /etc),
# a bare /tmp, mktemp templates (/tmp/name.XXXXXX), relative and home-relative paths, hidden tool state such as
# ~/.mu2efastsim, placeholders such as /path/to, and a bare quoted top-level prefix ("/pnfs/") that names no location,
# so code can keep refusing storage-namespace paths.

# User names, matched as whole alphanumeric tokens (case-insensitive) by SHA-256, so the names never appear here.
USER_NAME_SHA256 = frozenset({"4baf0263de48597ccb579ba119bb6d2f8fea4bba418326f375a4853462ab82fa",
                              "05436ea5391c8a9c46f09d1200e8669806543032d865c86649dfacc9293603f4"})
_TOKEN = re.compile(r"[A-Za-z0-9]+")


class UserNames:
    """search() like a compiled pattern: the first token whose SHA-256 is in ``hashes``, or None."""

    def __init__(self, hashes=USER_NAME_SHA256):
        self.hashes = hashes

    def search(self, line):
        for m in _TOKEN.finditer(line):
            if hashlib.sha256(m.group(0).lower().encode()).hexdigest() in self.hashes:
                return m
        return None


_PREFIXES = ("exp", "pnfs", "scratch", "sfs", "lus", "eagle", "grand", "flare", "home", "nashome", "Users", "project",
             "projects", "gpfs", "lustre", "wclustre", "global", "data1", "work")
# start of a path: right after file:// or not after a word, a dot, ~, }, $ or / (so ${VAR:-/exp/x} is still caught)
_START = r"(?:(?<=://)|(?<![\w.~}$/]))"
SITE_PATH = re.compile(_START + r"/(?:" + "|".join(_PREFIXES) + r")(?=/|\b)(?!/[\"'])")
TMP_DIR = re.compile(_START + r"/tmp/(?![\w.-]*\.XXXXXX)[A-Za-z0-9_$]")
HOME_DIR = re.compile(r"~/(?!\.)[A-Za-z0-9_]")
SITE_PATTERNS = (UserNames(), SITE_PATH, TMP_DIR, HOME_DIR)
_SKIP_DIRS = (".git", "__pycache__", ".pytest_cache", ".venv")


def _site_path_hit(line, patterns=SITE_PATTERNS):
    for rx in patterns:
        m = rx.search(line)
        if m:
            return m
    return None


def _repo_files(root, skip_top):
    try:
        # tracked plus untracked, not ignored, so a check before `git add` sees new files; submodules are not entered
        out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root,
                             check=True, capture_output=True).stdout
        names = [n for n in out.decode().split("\0") if n and n.split("/")[0] not in skip_top]
        if names:
            return [root / n for n in names]
    except (OSError, subprocess.CalledProcessError):
        pass
    # no git (e.g. inside an image): walk the tree, leaving out what the top-level .gitignore names (site.env, outputs)
    ignored = _gitignore_patterns(root)
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        top = here == root
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not (top and d in skip_top)
                       and not _ignored(here / d, root, ignored, True)]
        found.extend(here / f for f in filenames if f not in _SKIP_DIRS and not _ignored(here / f, root, ignored, False))
    return found


def _gitignore_patterns(root):
    try:
        lines = (root / ".gitignore").read_text().splitlines()
    except OSError:
        return []
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith(("#", "!"))]


def _ignored(path, root, patterns, is_dir):
    rel = path.relative_to(root).as_posix()
    for pat in patterns:
        if pat.endswith("/"):
            if not is_dir:
                continue
            pat = pat.rstrip("/")
        pat = pat.lstrip("/")
        if fnmatch.fnmatch(path.name, pat) or fnmatch.fnmatch(rel, pat):
            return True
    return False


def check_site_paths(root, skip_top=(), skip_files=()):
    """List "file:line: match in text" for every user name or site path in the repository at ``root``.

    ``skip_top``: top-level directories that are separate repositories (the umbrella's submodules). This module and
    ``skip_files`` are not scanned. Binary files are skipped.
    """
    root = Path(root).resolve()
    skip = {Path(__file__).resolve(), *(Path(f).resolve() for f in skip_files)}
    bad = []
    for path in _repo_files(root, tuple(skip_top)):
        if not path.is_file() or path.resolve() in skip:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue
        for lineno, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            m = _site_path_hit(line)
            if m:
                bad.append(f"{path.relative_to(root)}:{lineno}: {m.group(0)!r} in {line.strip()[:120]}")
    return bad


def site_path_self_test():
    """Assert that the checker's patterns catch site paths and leave public locations alone."""
    names = UserNames({hashlib.sha256(b"alice").hexdigest()})
    assert names.search("user ALICE here") and names.search("/data/alice/x") and names.search("nts.alice.tv2")
    assert not names.search("malice")
    patterns = (names, SITE_PATH, TMP_DIR, HOME_DIR)
    for s in ("/exp/mu2e/data/x", "cd /pnfs/mu2e", "'/scratch/abc'", "/lus/eagle/projects/x", "x=/home/u",
              "/Users/someone", "/project/area", "file:///pnfs/x", "/sfs/qumulo", "/nashome/u/x", "cd ~/mu2e/x",
              "out=/tmp/my_run", "Path('/tmp/thread-h')", "mkdir /tmp/$USER/x", "D=${M2FS_DATA:-/exp/mu2e/data/x}",
              'startswith("/pnfs/mu2e")', "sed 's|^/pnfs/mu2e|'"):
        assert _site_path_hit(s, patterns), s
    for s in ("/cvmfs/mu2e.opensciencegrid.org", "/usr/bin/env", "/path/to/scratch", "terminal", "determinant",
              "$HOME/exp/x", "${SCRATCH}/scratch/y", "relative/exp/z", "https://example.org/projects/x", "/tmp",
              "mktemp -d /tmp/offline-probe.XXXXXXXX", "/tmp|/tmp/*|/local|/localscratch", "/dev/shm/x",
              "~/.mu2efastsim/kit", 'startswith("/pnfs/")'):
        assert not _site_path_hit(s, patterns), s
    assert _ignored(Path("/r/site.env"), Path("/r"), ["site.env"], False)
    assert _ignored(Path("/r/a/plots"), Path("/r"), ["plots/"], True) and not _ignored(Path("/r/plots"), Path("/r"), ["plots/"], False)
    assert _ignored(Path("/r/x/y.root"), Path("/r"), ["*.root"], False) and not _ignored(Path("/r/x/y.py"), Path("/r"), ["*.root"], False)
