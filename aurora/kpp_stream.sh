#!/bin/bash
# Run from the laptop (it holds the ssh masters to a GPVM and to Aurora; the GPVM cannot reach Aurora).
# Submits kpp_reco_aurora.sh, streams the Mu2e-internal reference file GPVM -> Aurora compute-node tmpfs (never on
# Flare or the laptop disk), waits for the job, copies the outputs back to the GPVM data area and deletes them on Flare.
# usage: kpp_stream.sh [gpvm] [reference file on the GPVM]
set -euo pipefail
G=${1:-mu2egpvm03}
SRC=${2:-/exp/mu2e/data/users/ram2aq/crv_fastsim/reco_ref/crvreco_data.124155_000001.root}
FL=/lus/flare/projects/GenesisHackathonOct26/ram2aq/mu2efastsim
DEST=/exp/mu2e/data/users/ram2aq/crv_fastsim/reco_ref/checks_aurora
NAME=$(basename "$SRC")
J=$(ssh aurora "cd ~/m2fs && cp $FL/code/core/aurora/kpp_reco_aurora.sh . && qsub -v M2FS_FLARE=$FL,NAME=$NAME kpp_reco_aurora.sh" </dev/null)
JID=${J%%.*}; echo "job $JID"
until INBOX=$(ssh aurora "cat $FL/bench/kpp_inbox_$JID 2>/dev/null" </dev/null) && [ -n "$INBOX" ]; do sleep 30; done
read -r HOST L O <<<"$INBOX"; echo "node $HOST tmpfs $L out $O"
SUM=$(ssh "$G" "md5sum $SRC" </dev/null | cut -c1-32)
ssh "$G" "cat $SRC" </dev/null | ssh aurora "ssh -o StrictHostKeyChecking=no $HOST 'cat > $L/$NAME && md5sum $L/$NAME'" | tee /dev/stderr | grep -q "$SUM" \
  || { echo "checksum mismatch"; ssh aurora "qdel $JID" </dev/null; exit 4; }
ssh aurora "ssh $HOST 'touch $L/ready'; rm -f $FL/bench/kpp_inbox_$JID" </dev/null; echo "streamed $NAME ($SUM)"
while ssh aurora "qstat $JID >/dev/null 2>&1" </dev/null; do sleep 60; done
ssh aurora "tar -C $O -cf - ." </dev/null | ssh "$G" "mkdir -p $DEST/$JID && tar -C $DEST/$JID -xf - && ls $DEST/$JID"
ssh aurora "rm -rf $O" </dev/null; echo "outputs in $G:$DEST/$JID; removed $O"
