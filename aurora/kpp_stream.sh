#!/bin/bash
# Run from the laptop (it holds the ssh masters to a GPVM and to Aurora; the GPVM cannot reach Aurora).
# Submits kpp_reco_aurora.sh, streams the Mu2e-internal reference file GPVM -> Aurora compute-node tmpfs (never on
# Flare or the laptop disk), waits for the job, copies the outputs back to the GPVM data area and deletes them on Flare.
# usage: [EXTRA_V=VAR=value] [QSUB_OPTS="-q debug -l walltime=00:55:00"] kpp_stream.sh [gpvm] [job script] [reference files on the GPVM ...]
# Site settings (site.env.example): FASTSIMKIT_FNAL_HOST, AURORA_HOST (ssh host of Aurora), ALCF_PROJECT, M2FS_FLARE
# (work area on Flare) and the FNAL input dir (FASTSIMKIT_FNAL_DATA); the default reference file is
# <input dir>/reco_ref/crvreco_data.124155_000001.root.
set -euo pipefail
G=${1:-${FASTSIMKIT_FNAL_HOST:?set FASTSIMKIT_FNAL_HOST (FNAL host that holds the inputs) or pass it as the first argument}}
JOB=${2:-kpp_reco_aurora.sh}
shift $(( $# < 2 ? $# : 2 ))
D=${FASTSIMKIT_FNAL_DATA:-${CRVFS_DATA:-${M2FS_INPUTS:-${M2FS_DATA:?set FASTSIMKIT_FNAL_DATA (FNAL dir of the CRV inputs) or M2FS_INPUTS/M2FS_DATA}}/crv_fastsim}}
SRCS=("${@:-$D/reco_ref/crvreco_data.124155_000001.root}")
FL=${M2FS_FLARE:?set M2FS_FLARE (work area on Flare)}
PROJ=${ALCF_PROJECT:?set ALCF_PROJECT (ALCF allocation for qsub -A)}
AU=${AURORA_HOST:?set AURORA_HOST (ssh host of Aurora)}
DEST=$D/reco_ref/checks_aurora
NAME=$(basename "${SRCS[0]}")
J=$(ssh "$AU" "cd ~/m2fs && cp $FL/code/core/aurora/$JOB . && qsub -A $PROJ ${QSUB_OPTS:-} -v M2FS_FLARE=$FL,NAME=$NAME${EXTRA_V:+,$EXTRA_V} $JOB" </dev/null)
JID=${J%%.*}; echo "job $JID"
until INBOX=$(ssh "$AU" "cat $FL/bench/kpp_inbox_$JID 2>/dev/null" </dev/null) && [ -n "$INBOX" ]; do sleep 30; done
read -r HOST L O <<<"$INBOX"; echo "node $HOST tmpfs $L out $O"
for SRC in "${SRCS[@]}"; do
  N=$(basename "$SRC"); SUM=$(ssh "$G" "md5sum $SRC" </dev/null | cut -c1-32)
  ssh "$G" "cat $SRC" </dev/null | ssh "$AU" "ssh -o StrictHostKeyChecking=no $HOST 'cat > $L/$N && md5sum $L/$N'" | tee /dev/stderr | grep -q "$SUM" \
    || { echo "checksum mismatch $N"; ssh "$AU" "qdel $JID" </dev/null; exit 4; }
  echo "streamed $N ($SUM)"
done
ssh "$AU" "ssh $HOST 'touch $L/ready'; rm -f $FL/bench/kpp_inbox_$JID" </dev/null
while ssh "$AU" "qstat $JID >/dev/null 2>&1" </dev/null; do sleep 60; done
ssh "$AU" "tar -C $O -cf - ." </dev/null | ssh "$G" "mkdir -p $DEST/$JID && tar -C $DEST/$JID -xf - && ls $DEST/$JID"
ssh "$AU" "rm -rf $O" </dev/null; echo "outputs in $G:$DEST/$JID; removed $O"
