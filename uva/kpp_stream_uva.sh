#!/bin/bash
# Run from the laptop on the UVA network/VPN (it reaches both a Mu2e GPVM and UVA; neither reaches the other).
# Submits kpp_reco_uva.sh, streams the Mu2e-internal reference file GPVM -> laptop pipe -> UVA compute-node tmpfs
# (srun --overlap into the running job; never on /scratch or the laptop disk), waits for the job, copies the outputs
# (stats only) to the GPVM data area and deletes them on UVA.
# usage: kpp_stream_uva.sh [gpvm] [job script dir on UVA] [reference files on the GPVM ...]
set -euo pipefail
G=${1:-mu2egpvm03}
JD=${2:-mu2e/fastsim_uva}
shift $(( $# < 2 ? $# : 2 ))
SRCS=("${@:-/exp/mu2e/data/users/ram2aq/crv_fastsim/reco_ref/crvreco_data.124155_000001.root}")
U=${UVA_HOST:-uva-hpc}
F=/scratch/ram2aq/mu2efastsim
DEST=/exp/mu2e/data/users/ram2aq/crv_fastsim/reco_ref/checks_uva
NAME=$(basename "${SRCS[0]}")
J=$(ssh "$U" "mkdir -p $F/kpp && cd $JD && sbatch --parsable --export=ALL,NAME=$NAME kpp_reco_uva.sh" </dev/null)
JID=${J%%;*}; echo "job $JID"
until INBOX=$(ssh "$U" "cat $F/kpp/inbox_$JID 2>/dev/null" </dev/null) && [ -n "$INBOX" ]; do
  st=$(ssh "$U" "squeue -h -j $JID -o %T" </dev/null || true)
  [ -z "$st" ] && { echo "job $JID left the queue before asking for input"; exit 3; }
  sleep 30
done
read -r HOST L O <<<"$INBOX"; echo "node $HOST tmpfs $L out $O"
for SRC in "${SRCS[@]}"; do
  N=$(basename "$SRC"); SUM=$(ssh "$G" "md5sum $SRC" </dev/null | cut -c1-32)
  ssh "$G" "cat $SRC" </dev/null \
    | ssh "$U" "srun --jobid=$JID --overlap -N1 -n1 --mem=0 -w $HOST sh -c 'cat > $L/$N && md5sum $L/$N'" \
    | tee /dev/stderr | grep -q "$SUM" || { echo "checksum mismatch $N"; ssh "$U" "scancel $JID" </dev/null; exit 4; }
  echo "streamed $N ($SUM)"
done
ssh "$U" "srun --jobid=$JID --overlap -N1 -n1 --mem=0 -w $HOST touch $L/ready" </dev/null
while [ -n "$(ssh "$U" "squeue -h -j $JID -o %T" </dev/null)" ]; do sleep 60; done
ssh "$U" "sacct -j $JID -X -o JobID,Elapsed,State,ExitCode; tail -n 25 $O/run.log | cut -c1-400" </dev/null
ssh "$U" "tar -C $O -cf - ." </dev/null | ssh "$G" "mkdir -p $DEST/$JID && tar -C $DEST/$JID -xf - && ls $DEST/$JID"
ssh "$U" "rm -rf $O" </dev/null; echo "outputs in $G:$DEST/$JID; removed $O"
