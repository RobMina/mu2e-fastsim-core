#!/bin/bash
# usage: checkout_siblings.sh NAME... (core, crv, tracker, umbrella): shallow clone of each sibling repository into
# ./NAME at the branch of the same name as the one under test (CI_BRANCH) if it exists there, else main, so changes
# that span repositories are tested together. Existing directories are left alone.
set -eo pipefail
declare -A URL=([core]=mu2e-fastsim-core [crv]=mu2e-crv-fastsim [tracker]=mu2e-tracker-fastsim [umbrella]=mu2e-fastsim)
for name in "$@"; do
  [ -e "$name" ] && continue
  url=https://github.com/RobMina/${URL[$name]:?unknown repository $name}.git
  ref=main
  if [ -n "${CI_BRANCH:-}" ] && git ls-remote --exit-code --heads "$url" "$CI_BRANCH" >/dev/null 2>&1; then ref=$CI_BRANCH; fi
  git clone -q --depth 1 --branch "$ref" "$url" "$name"
  echo "$name: $ref $(git -C "$name" log -1 --format='%h %s')"
done
