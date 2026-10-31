#!/bin/sh
# usage: offline_data.sh DEST [offline_ref]: the public Offline files the fast chain reads, laid out for MU2E_SEARCH_PATH.
set -e
D=${1:?usage: offline_data.sh DEST [offline_ref]}; REF=${2:-v13_39_00}
U=https://raw.githubusercontent.com/Mu2e/Offline/$REF
for f in CRVResponse/data/singlePEWaveform_v3.txt Mu2eG4/geom/crv_counters_extracted_v04.txt; do
  mkdir -p "$D/Offline/$(dirname "$f")"
  [ -s "$D/Offline/$f" ] || curl -fsSL --retry 3 -o "$D/Offline/$f" "$U/$f"
done
