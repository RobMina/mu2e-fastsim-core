#!/bin/bash
# Run from the laptop on the UVA network/VPN (it reaches both a Mu2e GPVM and UVA; neither reaches the other).
# Submits kpp_reco_uva.sh, streams the Mu2e-internal reference file GPVM -> laptop pipe -> UVA compute-node tmpfs
# (srun --overlap into the running job; never on the cluster scratch or the laptop disk), waits for the job, copies the outputs
# (stats only) to the GPVM data area and deletes them on UVA.
# usage: [JOB=kpp_reco_uva.sh] [EXPORTS=VAR=v,VAR2=w] [SBATCH_OPTS=...] kpp_stream_uva.sh [gpvm] [job dir on UVA] [files on the GPVM ...]
# (EXPORTS is passed to sbatch --export: commas separate variables, so a value cannot contain one; RUNS="1 4" works)
# Site settings (site.env.example): FASTSIMKIT_FNAL_HOST, FASTSIMKIT_SLURM_HOST, FASTSIMKIT_SLURM_ACCOUNT, FASTSIMKIT_SCRATCH
# and the FNAL input dir (FASTSIMKIT_FNAL_DATA); the default reference file is <input dir>/reco_ref/crvreco_data.124155_000001.root.
set -euo pipefail
G=${1:-${FASTSIMKIT_FNAL_HOST:?set FASTSIMKIT_FNAL_HOST (FNAL host that holds the inputs) or pass it as the first argument}}
JD=${2:-mu2e/fastsim_uva}
shift $(( $# < 2 ? $# : 2 ))
D=${FASTSIMKIT_FNAL_DATA:-${CRVFS_DATA:-${M2FS_INPUTS:-${M2FS_DATA:?set FASTSIMKIT_FNAL_DATA (FNAL dir of the CRV inputs) or M2FS_INPUTS/M2FS_DATA}}/crv_fastsim}}
SRCS=("${@:-$D/reco_ref/crvreco_data.124155_000001.root}")
U=${UVA_HOST:-${FASTSIMKIT_SLURM_HOST:?set FASTSIMKIT_SLURM_HOST (ssh host of the Slurm cluster)}}
F=${FASTSIMKIT_SCRATCH:?set FASTSIMKIT_SCRATCH (work area on the cluster scratch)}
ACCT=${FASTSIMKIT_SLURM_ACCOUNT:?set FASTSIMKIT_SLURM_ACCOUNT (Slurm account)}
DEST=$D/reco_ref/checks_uva
NAME=$(basename "${SRCS[0]}")
JOB=${JOB:-kpp_reco_uva.sh}
J=$(ssh "$U" "mkdir -p $F/kpp && cd $JD && sbatch --parsable -A $ACCT -o $F/kpp/slurm-%j.out ${SBATCH_OPTS:-} --export='ALL,FASTSIMKIT_SCRATCH=$F,NAME=$NAME${EXPORTS:+,$EXPORTS}' $JOB" </dev/null)
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
ssh "$U" "cp $F/kpp/slurm-$JID.out $O/slurm.out 2>/dev/null; rm -f $F/kpp/slurm-$JID.out" </dev/null
ssh "$U" "sacct -j $JID -X -o JobID,Elapsed,State,ExitCode; tail -n 25 $O/run.log | cut -c1-400" </dev/null
ssh "$U" "tar -C $O -cf - ." </dev/null | ssh "$G" "mkdir -p $DEST/$JID && tar -C $DEST/$JID -xf - && ls $DEST/$JID"
ssh "$U" "rm -rf $O" </dev/null; echo "outputs in $G:$DEST/$JID; removed $O"
