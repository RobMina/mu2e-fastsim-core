"""Exercise the private extractor's Apptainer dependency-discovery protocol without building."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_recipe_staging_and_shell_syntax():
    root = Path(__file__).resolve().parents[1]
    recipe = (root / "native/cry/cry-overlay.def").read_text()
    assert "/".join(("", "tmp", "cry-build")) not in recipe   # no build staging in a shared tmp dir
    files = recipe.split("%files\n", 1)[1].split("%post", 1)[0]
    for row in files.strip().splitlines():
        source, dest = row.split()
        assert (root / source).is_file()
        assert dest.startswith("/opt/cry-build/")
    sections = recipe.split("%post\n")[1:]
    assert len(sections) == 2
    assert sections[0].index("test -r cry_shim.cc") < sections[0].index("apt-get")
    for section in sections:
        script = section.split("\nBootstrap:", 1)[0].split("\n%", 1)[0]
        subprocess.run(["bash", "-n"], input=script, text=True, check=True)


@pytest.mark.skipif(not shutil.which("bash"), reason="requires bash")
def test_prepare_wrapper_protocol(tmp_path):
    root = Path(__file__).resolve().parents[1]
    tools = tmp_path / "tools"
    tools.mkdir()
    config = tmp_path / "site" / "apptainer"
    config.mkdir(parents=True)
    (config / "apptainer.conf").write_text("mount proc = yes\nbinary path = /site/tools:$PATH\n")
    apptainer = tools / "apptainer"
    apptainer.write_text(f"#!/bin/sh\nprintf 'SYSCONFDIR={config.parent}\\n'\n")
    real = tools / "unsquashfs"
    real.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    for path in (apptainer, real):
        path.chmod(0o755)
    env = {**os.environ, "PATH": f"{tools}:{os.environ['PATH']}"}
    dest = tmp_path / "private"
    subprocess.run(["bash", str(root / "native/cry/prepare-extraction.sh"), str(dest)],
                   check=True, env=env, capture_output=True, text=True)
    wrapper = dest / "bin/unsquashfs"
    probe = subprocess.run([str(wrapper)], env={"_WRAPPER_EXEC_CMD": "echo"},
                           check=True, capture_output=True, text=True)
    assert probe.stdout == f"{real}\n"
    assert probe.stderr == ""
    deps = subprocess.run([str(wrapper)], env={"_WRAPPER_EXEC_CMD": "/bin/echo --list"},
                          check=True, capture_output=True, text=True)
    assert deps.stdout == f"--list {real}\n"
    assert deps.stderr == ""
    extract = subprocess.run([str(wrapper), "-version"], env={}, check=True,
                             capture_output=True, text=True)
    assert extract.stdout.splitlines() == ["-processors", "4", "-data-queue", "64",
                                           "-frag-queue", "64", "-version"]
    assert "capped at 4" in extract.stderr
    private = (dest / "apptainer.conf").read_text()
    assert "mount proc = yes" in private
    assert "binary path = /site/tools" not in private
    assert f"binary path = {dest}/bin:$PATH:" in private
    repeated = subprocess.run(["bash", str(root / "native/cry/prepare-extraction.sh"), str(dest)],
                              env=env, capture_output=True)
    assert repeated.returncode != 0
