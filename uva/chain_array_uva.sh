#!/bin/bash
# Run from the laptop on the UVA network. The fast CRV chain as N independent small Slurm jobs (default: 1 GPU, 16
# cores each, any A100) instead of one whole node, so each job clears the queue on its own. Each job is
# chain_node_uva.sh with NPROC=1; job k gets gen files k*FPP .. k*FPP+FPP-1 (cycled). One umbrella checkout at $REF is
# prepared on the login node and shared (CODE_READY=1). Inputs are streamed GPVM -> laptop -> each job's node tmpfs
# when it starts (srun --overlap), never stored on UVA; outputs (stats only) go back to the GPVM and are removed.
# Then throughput_uva.py: accepted events per GPU-hour, per SU, and per hour of campaign wall time.
# usage: [N=8] [FPP=4] [GRES=gpu:a100:1] [CONSTRAINT=a100] [CPUS=16] [MEM=110G] [WALL=00:45:00] [REF=main] \
#        [T1B=strict_v3|strict_v2|legacy] chain_array_uva.sh [gpvm]      (T1B: tier-1b asset set, crvfastsim.tier1b_assets)
set -uo pipefail
G=${1:-mu2egpvm03}
N=${N:-8}; FPP=${FPP:-4}; GRES=${GRES:-gpu:a100:1}; CONSTRAINT=${CONSTRAINT:-a100}; CPUS=${CPUS:-16}
MEM=${MEM:-110G}; WALL=${WALL:-00:45:00}; REF=${REF:-main}; MAXEV=${MAXEV:-}; CRV_REF=${CRV_REF:-}   # CRV_REF: a crv branch to test
U=${UVA_HOST:-uva-hpc}; F=/scratch/ram2aq/mu2efastsim; JD=mu2e/fastsim_uva
D=/exp/mu2e/data/users/ram2aq/crv_fastsim
T1B=${T1B:-strict_v3}
case $T1B in
  strict_v3|strict_v2) TV=${T1B#strict_}
                     LIBSET=("$D/tier1b_strict_v2/library.root" "$D/tier1b_strict_v2/topology_sidecar_strict_v2_v2.npz"
                     "$D/tier1b_strict_v2/topology_sidecar_strict_v2_v2.npz.json" "$D/tier1b_strict_v2/templates_$TV.npz"
                     "$D/tier1b_strict_v2/templates_$TV.npz.json") ;;
  legacy) LIBSET=("$D/tier1b_v7/lib_prod_as_20files.root") ;;
  *) echo "T1B must be strict_v3, strict_v2 or legacy"; exit 2 ;;
esac
COMMON=("${LIBSET[@]}" "$D/geometry/crvcalib_sim_best_v1_5_run1400.csv"
        "$D/geometry/crvphoton_sim_best_v1_5_run1400.csv" "$D/geometry/mu2e_extracted_v04.gdml")
NGEN=${NGEN:-20}
TAG=array_$(date +%Y%m%d_%H%M%S)
LOG=${LOGDIR:-$HOME/mu2e/scratch_uva}/$TAG; mkdir -p "$LOG"
DEST=$D/reco_ref/checks_uva/$TAG
CODE=$F/chain_code_$REF${CRV_REF:+_crv-$CRV_REF}
echo "tag $TAG: N=$N FPP=$FPP tier1b=$T1B gres=$GRES constraint=$CONSTRAINT cpus=$CPUS mem=$MEM wall=$WALL ref=$REF crv=${CRV_REF:-pinned}; log $LOG"

# one checkout for all jobs (login node: git only, no compute)
ssh "$U" "set -e; if [ -d $CODE/.git ]; then git -C $CODE fetch -q origin; git -C $CODE checkout -q $REF; git -C $CODE reset -q --hard origin/$REF; \
  else git clone -q https://github.com/RobMina/mu2e-fastsim.git $CODE; git -C $CODE checkout -q $REF; fi; \
  git -C $CODE submodule update -q --init --recursive; \
  if [ -n '$CRV_REF' ]; then git -C $CODE/crv fetch -q origin $CRV_REF && git -C $CODE/crv checkout -q FETCH_HEAD; fi; \
  git -C $CODE log -1 --format='code %h %s' | cut -c1-100; git -C $CODE/crv log -1 --format='crv %h %s' | cut -c1-100" </dev/null || exit 2

JIDS=()
for k in $(seq 0 $((N-1))); do
  J=$(ssh "$U" "cd $JD && sbatch --parsable -J m2fs_arr$k --gres=$GRES -C $CONSTRAINT -c $CPUS --mem=$MEM -t $WALL \
      --export='ALL,NPROC=1,RUNS=$FPP,SEED_BASE=$((20261021 + 1000*k)),CODE_DIR=$CODE,CODE_READY=1,T1B=$T1B${MAXEV:+,MAXEV=$MAXEV}' chain_node_uva.sh" </dev/null)
  JIDS+=("${J%%;*}"); echo "job $k -> ${J%%;*}"
done
printf "%s\n" "${JIDS[@]}" > "$LOG/jobids.txt"

stream_one() {   # $1 job index, $2 job id: wait for the inbox, stream this job's files, mark ready, wait for the end
  local k=$1 JID=$2 INBOX HOST L O st files i
  until INBOX=$(ssh "$U" "cat $F/kpp/inbox_$JID 2>/dev/null" </dev/null) && [ -n "$INBOX" ]; do
    st=$(ssh "$U" "squeue -h -j $JID -o %T" </dev/null || true)
    [ -z "$st" ] && { echo "[$JID] left the queue before asking for input: $(ssh "$U" "sacct -n -X -j $JID -o State,ExitCode; tail -3 $F/kpp/slurm-$JID.out" </dev/null | tr '\n' ' ' | cut -c1-300)"; return 3; }
    sleep 30
  done
  read -r HOST L O <<<"$INBOX"; echo "[$JID] node $HOST"
  files=("${COMMON[@]}")
  for i in $(seq 0 $((FPP-1))); do files+=("$(printf "$D/g4ref_v1_tables/gen.001400_%08d.root" $(( (k*FPP+i) % NGEN + 1 )))"); done
  for SRC in "${files[@]}"; do
    local N1 SUM; N1=$(basename "$SRC"); SUM=$(ssh "$G" "md5sum $SRC" </dev/null | cut -c1-32)
    ssh "$G" "cat $SRC" </dev/null | ssh "$U" "srun --jobid=$JID --overlap -N1 -n1 --mem=0 -w $HOST sh -c 'cat > $L/$N1 && md5sum $L/$N1'" \
      | grep -q "$SUM" || { echo "[$JID] checksum mismatch $N1"; ssh "$U" "scancel $JID" </dev/null; return 4; }
  done
  ssh "$U" "srun --jobid=$JID --overlap -N1 -n1 --mem=0 -w $HOST touch $L/ready" </dev/null
  echo "[$JID] streamed ${#files[@]} files"
  while [ -n "$(ssh "$U" "squeue -h -j $JID -o %T" </dev/null)" ]; do sleep 60; done
  ssh "$U" "cp $F/kpp/slurm-$JID.out $O/slurm.out 2>/dev/null; rm -f $F/kpp/slurm-$JID.out $F/kpp/inbox_$JID; tar -C $O -cf - ." </dev/null \
    | ssh "$G" "mkdir -p $DEST/$JID && tar -C $DEST/$JID -xf -"
  ssh "$U" "rm -rf $O" </dev/null
  echo "[$JID] done: $(ssh "$U" "sacct -n -X -j $JID -o State,Elapsed" </dev/null | xargs)"
}
pids=()
for k in $(seq 0 $((N-1))); do stream_one $k "${JIDS[$k]}" > "$LOG/stream_$k.log" 2>&1 & pids+=($!); sleep 2; done
for p in "${pids[@]}"; do wait $p; done
cat "$LOG"/stream_*.log

# accounting: Slurm times and MAM SU per job (MAM posts the charge a little after the job ends)
sleep 60
ssh "$U" "for j in ${JIDS[*]}; do \
  s=\$(sacct -n -X -P -j \$j -o Start,End,ElapsedRaw,NodeList,AllocTRES,Partition | head -1); \
  su=\$(mam-list-transactions -a ldmxuva -s \$(date +%F -d yesterday) 2>/dev/null | awk -v j=\$j '\$6==j && \$9>0 {print \$9}' | head -1); \
  echo \"\$j|\$s|\$su\"; done" </dev/null > "$LOG/acct.txt"
ssh "$G" "cd $DEST && for j in ${JIDS[*]}; do w=\$(grep -ho 'input wait [0-9]*' \$j/run.log 2>/dev/null | awk '{print \$3}'); echo \"\$j \${w:-0}\"; done" </dev/null > "$LOG/waits.txt"
python3 - "$LOG" <<'EOF'
import sys, datetime as dt
log = sys.argv[1]
waits = dict(l.split() for l in open(f"{log}/waits.txt") if l.strip())
rows = ["jobid\tstart\tend\telapsed_s\tgpus\tgpu_type\tnode\tsu\tinput_wait_s"]
for line in open(f"{log}/acct.txt"):
    jid, start, end, el, node, tres, part, su = (line.strip().split("|") + [""] * 8)[:8]
    if not start or start == "Unknown":
        continue
    ts = lambda s: dt.datetime.fromisoformat(s).timestamp()
    gpus = next((int(x.split("=")[1]) for x in tres.split(",") if x.startswith("gres/gpu=")), 1)
    gtype = part.replace("gpu-", "") or "gpu"   # partition name carries the model and memory (gpu-a100-40, gpu-a100-80)
    rows.append(f"{jid}\t{ts(start)}\t{ts(end)}\t{el}\t{gpus}\t{gtype}\t{node}\t{su or 0}\t{waits.get(jid, 0)}")
open(f"{log}/jobs.tsv", "w").write("\n".join(rows) + "\n")
print(f"{len(rows) - 1} jobs with accounting -> {log}/jobs.tsv")
EOF
scp -q "$LOG/jobs.tsv" "$G:$DEST/jobs.tsv"
ssh "$G" "cd $DEST && python3 /exp/mu2e/app/users/ram2aq/analysis/mu2e-fastsim/core/uva/throughput_uva.py jobs.tsv */fpp$FPP/shard_0.json --out throughput.json" </dev/null \
  | tee "$LOG/throughput.txt"
echo "outputs in $G:$DEST"
