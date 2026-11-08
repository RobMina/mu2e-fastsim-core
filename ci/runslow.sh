#!/bin/bash
# Decide the test tier for CI and write "args=..." to $GITHUB_OUTPUT: "--runslow" on the nightly schedule, on a manual
# run with runslow, and on a pull request that changes a path matching a line of ci/runslow_paths.txt (contracts
# whose tests run only in the slow tier); empty otherwise (the per-PR tier).
set -eo pipefail
args=""
case "${EVENT:-}" in
  schedule) args=--runslow ;;
  workflow_dispatch) [ "${DISPATCH_RUNSLOW:-true}" = true ] && args=--runslow ;;
  pull_request)
    if [ -n "${BASE_SHA:-}" ] && [ -f ci/runslow_paths.txt ]; then
      pattern=$(grep -v -E '^\s*(#|$)' ci/runslow_paths.txt | paste -sd'|' -)
      if [ -n "$pattern" ] && git diff --name-only "$BASE_SHA"...HEAD | grep -E -q "$pattern"; then
        args=--runslow
        echo "changed files match ci/runslow_paths.txt: running the slow tier too"
      fi
    fi ;;
esac
echo "tier: ${args:-per-PR}"
echo "args=$args" >> "${GITHUB_OUTPUT:-/dev/null}"
