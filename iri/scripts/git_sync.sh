#!/bin/bash
# Clone or update the public mu2e-fastsim umbrella repo and its submodules on Eagle (HTTPS through the ALCF proxy).
# usage (positional args via make_inline_job.py --args): git_sync.sh [fresh]   ("fresh" removes the clone first)
CODE=${M2FS_EAGLE:?}/code
URL=https://github.com/RobMina/mu2e-fastsim.git
export https_proxy=http://proxy.alcf.anl.gov:3128 http_proxy=http://proxy.alcf.anl.gov:3128
if [ "${1:-}" = fresh ] && [ -d "$CODE" ]; then
  rm -rf "$CODE" && echo "removed $CODE"
fi
if [ -d "$CODE/.git" ]; then
  git -C "$CODE" pull --ff-only 2>&1 | tail -3
  git -C "$CODE" submodule update --init --recursive 2>&1 | tail -3
else
  git clone --recurse-submodules "$URL" "$CODE" 2>&1 | tail -3
fi
git -C "$CODE" log -1 --format='HEAD %h %ad %s' --date=iso 2>&1
git -C "$CODE" submodule status 2>&1
echo "exit=0"
