"""Select the test files a change can affect, from the static import graph of one or more repositories.

    python -m mu2efastsim.testselect [--repo DIR] [--base REF] [--also DIR[:REF] ...] [--run [--isolate]] [-- PYTEST_ARGS]

Changed files are those between the merge base of ``--base`` (default ``origin/main``) and the working tree,
tracked or untracked. A test file is selected when it changed, or when it imports (directly or through other
modules, including imports inside functions) a changed module. ``--also`` adds a sibling repository whose own
changes count too (for example ``--also ../core`` from the CRV repository, so a core change selects the CRV tests
that use it); every repository given is scanned to resolve imports.

Conservative rules: a changed package data file (``crvfastsim/configs/x.json``) counts as a change of its package;
a changed ``conftest.py`` selects the tests in its directory; ``pyproject.toml``, ``setup.cfg`` or ``pytest.ini``
selects every test. Documentation and non-Python files outside packages select nothing. Imports made from strings
(``importlib``, entry points) are invisible: use ``--all`` when a change is about registration or plugins.

``--run`` runs the selection with pytest in the repository root (PYTHONPATH as the caller set it). ``--isolate``
runs one pytest process per test file, which bounds peak memory to the heaviest file instead of the whole suite.
"""
import argparse
import ast
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ALL_TRIGGERS = {"pyproject.toml", "setup.cfg", "pytest.ini", "tox.ini"}


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def changed_files(repo, base):
    """Paths (relative to the repository root) changed since the merge base with ``base``, plus uncommitted ones."""
    merge_base = _git(repo, "merge-base", base, "HEAD").strip()
    names = set(_git(repo, "diff", "--name-only", merge_base).split())
    names |= set(_git(repo, "ls-files", "--others", "--exclude-standard").split())
    return sorted(names)


def python_files(repo):
    names = _git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "--", "*.py").split()
    return [n for n in names if (Path(repo) / n).is_file()]


def module_name(repo, rel):
    """Dotted import name of ``rel``: the path below the nearest ancestor directory without ``__init__.py``."""
    path = Path(repo) / rel
    parts = [path.stem] if path.name != "__init__.py" else []
    d = path.parent
    while (d / "__init__.py").is_file() and d != Path(repo).parent:
        parts.insert(0, d.name)
        d = d.parent
    return ".".join(parts), (path.name == "__init__.py")


def imports(source, module, is_package):
    """Every module name an import statement in ``source`` can load (absolute names; parents included)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    package = module if is_package else module.rpartition(".")[0]
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".") if package else []
                base = base[:len(base) - (node.level - 1)] if node.level > 1 else base
                root = ".".join(base + ([node.module] if node.module else []))
            else:
                root = node.module or ""
            names = [root] + [f"{root}.{a.name}" if root else a.name for a in node.names]
        else:
            continue
        for name in names:
            bits = name.split(".")
            found.update(".".join(bits[:i]) for i in range(1, len(bits) + 1))
    return found


class Graph:
    def __init__(self, repos):
        self.path = {}        # (repo, rel) -> module name
        self.where = defaultdict(list)   # module name -> [(repo, rel)]
        self.deps = {}        # (repo, rel) -> set of imported module names
        for repo in repos:
            for rel in python_files(repo):
                name, is_pkg = module_name(repo, rel)
                self.path[(repo, rel)] = name
                self.where[name].append((repo, rel))
                self.deps[(repo, rel)] = imports((Path(repo) / rel).read_text(errors="replace"), name, is_pkg)

    def resolve(self, repo, name):
        nodes = self.where.get(name, [])
        same = [n for n in nodes if n[0] == repo]
        return same or nodes

    def dependents(self, changed):
        """All nodes that import, transitively, any node in ``changed``."""
        rdeps = defaultdict(set)
        for node, names in self.deps.items():
            for name in names:
                for target in self.resolve(node[0], name):
                    if target != node:
                        rdeps[target].add(node)
        seen, todo = set(changed), list(changed)
        while todo:
            for parent in rdeps[todo.pop()]:
                if parent not in seen:
                    seen.add(parent)
                    todo.append(parent)
        return seen


def is_test(rel):
    p = Path(rel)
    return p.name.startswith("test_") and p.suffix == ".py" and "tests" in p.parts


def select(repo, base="origin/main", also=(), run_all=False):
    """Return (sorted test paths of ``repo``, reasons dict, changed files per repository)."""
    repo = str(Path(repo).resolve())
    others = [(str(Path(d).resolve()), ref or base) for d, ref in also]
    graph = Graph([repo] + [d for d, _ in others])
    tests = sorted(rel for (r, rel) in graph.path if r == repo and is_test(rel))
    changes = {repo: changed_files(repo, base)}
    for d, ref in others:
        changes[d] = changed_files(d, ref)
    if run_all or any(Path(f).name in ALL_TRIGGERS for f in changes[repo]):
        return tests, {t: "all" for t in tests}, changes
    seeds, reasons, direct = set(), {}, set()
    for r, files in changes.items():
        for f in files:
            p = Path(f)
            if (r, f) in graph.path:
                seeds.add((r, f))
                if r == repo and is_test(f):
                    direct.add(f)
            elif p.name == "conftest.py" and r == repo:
                direct.update(t for t in tests if Path(t).parent == p.parent)
            elif p.suffix != ".py" and (Path(r) / p.parent / "__init__.py").is_file():
                init = str(p.parent / "__init__.py")
                if (r, init) in graph.path:   # package data: count as a change of the package
                    seeds.add((r, init))
    for t in direct:
        reasons[t] = "changed"
    for node in graph.dependents(seeds):
        if node[0] == repo and is_test(node[1]):
            reasons.setdefault(node[1], "imports a changed module")
    return sorted(reasons), reasons, changes


def _run(repo, tests, pytest_args, isolate):
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *pytest_args]
    if not isolate:
        return subprocess.call(cmd + tests, cwd=repo)
    failed = []
    for t in tests:
        print(f"== {t}", flush=True)
        if subprocess.call(cmd + [t], cwd=repo) not in (0, 5):
            failed.append(t)
    print(f"{len(tests) - len(failed)}/{len(tests)} files passed" + (f"; failed: {' '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default=".", help="repository whose tests are selected (default: current directory)")
    ap.add_argument("--base", default="origin/main", help="compare against the merge base with this ref")
    ap.add_argument("--also", action="append", default=[], metavar="DIR[:REF]",
                    help="sibling repository whose changes also count (repeatable)")
    ap.add_argument("--all", action="store_true", help="select every test")
    ap.add_argument("--why", action="store_true", help="print the reason for each selected file")
    ap.add_argument("--run", action="store_true", help="run the selection with pytest")
    ap.add_argument("--isolate", action="store_true", help="with --run: one pytest process per test file")
    a = ap.parse_args(argv)
    also = [tuple(x.split(":", 1)) if ":" in x else (x, None) for x in a.also]
    tests, reasons, changes = select(a.repo, a.base, also, a.all)
    n = sum(len(v) for v in changes.values())
    print(f"# {n} changed file(s); {len(tests)} test file(s) selected", file=sys.stderr)
    if not a.run:
        for t in tests:
            print(f"{t}\t{reasons[t]}" if a.why else t)
        return 0
    if not tests:
        print("nothing to run", file=sys.stderr)
        return 0
    return _run(os.path.abspath(a.repo), tests, extra, a.isolate)


if __name__ == "__main__":
    sys.exit(main())
