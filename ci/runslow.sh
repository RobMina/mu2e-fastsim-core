#!/bin/bash
# Decide the test tier for CI and write "args=..." to $GITHUB_OUTPUT:
#   "--runweekly -m weekly" on the weekly schedule and on a manual run with weekly (only the weekly validations);
#   "--runslow" on the nightly schedule, on a manual run with runslow, and on a pull request that changes a path
#   matching a line of ci/runslow_paths.txt (contracts whose tests run only in the slow tier);
#   empty otherwise (the per-PR tier).
set -eo pipefail
args=""
# The weekly schedule (cron "11 3 * * 0" in core's tests.yml) and a manual run with weekly run only the weekly tier.
WEEKLY_CRON="11 3 * * 0"
case "${EVENT:-}" in
  schedule)
    if [ "${SCHEDULE:-}" = "$WEEKLY_CRON" ]; then args="--runweekly -m weekly"; else args=--runslow; fi ;;
  workflow_dispatch)
    if [ "${DISPATCH_WEEKLY:-false}" = true ]; then args="--runweekly -m weekly"
    elif [ "${DISPATCH_RUNSLOW:-true}" = true ]; then args=--runslow; fi ;;
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
