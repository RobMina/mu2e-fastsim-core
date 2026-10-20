#!/bin/bash
# ESnet East pod (public ghcr.io/robmina/mu2e-fastsim image): pytest, CRV response bench with the REAL v6_0 tables
# and the synthetic reco chain on CPU, R5/P6 repro. Tables: CVMFS Stratum-1 -> pod-local /tmp, deleted on exit (never EFS).
# Env (optional): NROWS, NSIPMS, NEV, GPU=1 (JAX on the GPU instead of CPU).
O=$PWD; NROWS=${NROWS:-200000}; NSIPMS=${NSIPMS:-100000}; NEV=${NEV:-20000}
echo "== start $(date -u +%FT%TZ) host $(hostname) job ${JOB_ID:-none} out $O"; cat /opt/mu2e-fastsim/VERSION
echo "cpus $(nproc) quota $(cat /sys/fs/cgroup/cpu.max 2>/dev/null) mem $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
grep -m1 'model name' /proc/cpuinfo; [ "${GPU:-0}" = 1 ] || export JAX_PLATFORMS=cpu
L=/tmp/crvfs_lookup_$$; mkdir -p $L; chmod 700 $L
trap 'rm -rf $L; echo "deleted $L ($(ls $L 2>/dev/null | wc -l) files left)"' EXIT
trap 'exit 143' TERM INT
for f in LookupTable_3200_0 photonMap.root; do
  python -m mu2efastsim.cvmfsget mu2e.opensciencegrid.org /DataFiles/CRVConditions/v6_0/$f $L/$f || { echo "fetch $f failed"; exit 3; }
done
ls -la $L; export CRVFS_LOOKUP_DIR=$L
python -c "import jax; print('jax', jax.__version__, jax.devices())"
cd /opt/mu2e-fastsim
echo; echo "==== pytest per package (shared test basenames); tracker last (one test needs > 7 GB)"
for p in core crv tracker; do echo "-- $p: $(python -m pytest -q -rfE -p no:cacheprovider $p/tests 2>&1 | tail -1)"; done
B="python -m crvfastsim.response.bench --repeat 3"
echo; echo "==== photons, real table 3200_0 ($NROWS rows; numpy 20k)"
$B --table 3200_0 --rows $NROWS --numpy-rows 20000 --seed 20261020 2>&1 | tail -1
echo; echo "==== SiPM charges, real photon map ($NSIPMS SiPMs; numpy 20k)"
$B --stage sipm --photon-map $L/photonMap.root --sipms $NSIPMS --numpy-sipms 20000 --seed 20261120 2>&1 | tail -1
echo; echo "==== reco chain, synthetic ($NEV events, JAX)"
python -m crvfastsim reco-bench synthetic --stage chain --max-events $NEV 2>&1 | tail -3
echo; echo "==== R5/P6 reco-repro, synthetic (5000 events)"
python -m crvfastsim reco-repro synthetic --max-events 5000 --outdir /tmp/repro_$$ 2>&1 | tail -3
python -c "import json,glob;d=json.load(open(glob.glob('/tmp/repro_$$/*.json')[0]));print({k:d[k] for k in ('backend','device','pass_bit_identical','same_seed_identical','fingerprint')})"
cp /tmp/repro_$$/* $O/ 2>/dev/null; rm -rf /tmp/repro_$$   # synthetic only
echo "== done $(date -u +%FT%TZ)"; echo exit=0
