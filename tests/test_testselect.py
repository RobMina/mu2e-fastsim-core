"""testselect: changed modules select exactly the test files that import them, directly or transitively."""
import subprocess

from mu2efastsim import testselect as TS


def _repo(tmp, files):
    for rel, text in files.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    git = lambda *a: subprocess.run(["git", "-C", str(tmp), *a], check=True, capture_output=True)
    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    return tmp


FILES = {
    "pkg/__init__.py": "",
    "pkg/a.py": "X = 1\n",
    "pkg/b.py": "from .a import X\n",
    "pkg/c.py": "def f():\n    from pkg import b\n    return b\n",
    "pkg/data/table.json": "{}\n",
    "pkg/data/__init__.py": "",
    "pkg/d.py": "Y = 2\n",
    "tests/test_b.py": "import pkg.b\n",
    "tests/test_c.py": "from pkg.c import f\n",
    "tests/test_d.py": "from pkg import d\n",
    "tests/helper_x.py": "import pkg.d\n",
    "tests/test_uses_helper.py": "import helper_x\n",
    "README.md": "x\n",
}


def test_transitive_relative_and_function_level_imports(tmp_path):
    repo = _repo(tmp_path, FILES)
    (repo / "pkg/a.py").write_text("X = 3\n")
    tests, reasons, _ = TS.select(repo, base="HEAD")
    assert tests == ["tests/test_b.py", "tests/test_c.py"]


def test_helper_modules_in_tests_and_untracked_files(tmp_path):
    repo = _repo(tmp_path, FILES)
    (repo / "pkg/d.py").write_text("Y = 4\n")
    (repo / "tests/test_new.py").write_text("X = 0\n")
    tests, reasons, _ = TS.select(repo, base="HEAD")
    assert tests == ["tests/test_d.py", "tests/test_new.py", "tests/test_uses_helper.py"]
    assert reasons["tests/test_new.py"] == "changed"


def test_docs_select_nothing_and_package_data_selects_importers(tmp_path):
    repo = _repo(tmp_path, FILES)
    (repo / "README.md").write_text("y\n")
    assert TS.select(repo, base="HEAD")[0] == []
    (repo / "pkg/data/table.json").write_text("{\"a\": 1}\n")
    assert TS.select(repo, base="HEAD")[0] == []   # nothing imports pkg.data
    (repo / "pkg/b.py").write_text("from .a import X\nfrom . import data\n")
    tests = TS.select(repo, base="HEAD")[0]
    assert "tests/test_b.py" in tests and "tests/test_c.py" in tests


def test_config_change_selects_everything(tmp_path):
    repo = _repo(tmp_path, {**FILES, "pyproject.toml": "[project]\n"})
    (repo / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    tests, reasons, _ = TS.select(repo, base="HEAD")
    assert len(tests) == 4 and set(reasons.values()) == {"all"}


def test_sibling_repository_changes(tmp_path):
    core = _repo(tmp_path / "core", {"lib/__init__.py": "", "lib/m.py": "Z = 1\n", "lib/n.py": "W = 1\n"})
    user = _repo(tmp_path / "user", {"app/__init__.py": "", "app/u.py": "from lib.m import Z\n",
                                     "tests/test_u.py": "import app.u\n", "tests/test_v.py": "import lib.n\n"})
    (core / "lib/m.py").write_text("Z = 2\n")
    tests, _, changes = TS.select(user, base="HEAD", also=[(core, "HEAD")])
    assert tests == ["tests/test_u.py"] and changes[str(core.resolve())] == ["lib/m.py"]
