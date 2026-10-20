#!/bin/bash
#PBS -N m2fs_resp
#PBS -A GenesisHackathonOct26
#PBS -q debug
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:55:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o m2fs_resp.out
# Aurora: CRV fast response (photons + SiPM charges) with the REAL v6_0 lookup table and photon map, one PVC tile and
# then all 12 tiles (one process per tile). Tables come from the Mu2e CVMFS Stratum-1 over HTTP into tmpfs /tmp and
# are deleted on exit (Mu2e-internal: never on Flare). Needs the venv of probe_aurora.sh.
# usage: qsub [-v M2FS_FLARE=...,N=rows_per_tile,NS=sipms_per_tile] response_aurora.sh
PROJ=${PBS_ACCOUNT:-GenesisHackathonOct26}
F=${M2FS_FLARE:-/lus/flare/projects/$PROJ/$USER/mu2efastsim}
FL=$F; V=$FL/venv; CODE=$FL/code; OFFD=$FL/offline_data; N=${N:-1000000}; NS=${NS:-500000}
O=$FL/bench/response_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}; mkdir -p $O
exec > >(tee "$O/run.log") 2>&1
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy no_proxy="localhost,*.alcf.anl.gov"
cd $CODE && git pull -q --ff-only && git submodule update -q --init --recursive
echo "host=$(hostname) job=${PBS_JOBID:-none} out=$O"; git log -1 --format="umbrella %h %s"; git submodule status
export LD_LIBRARY_PATH=$V/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} MU2E_SEARCH_PATH=$OFFD
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg ZE_FLAT_DEVICE_HIERARCHY=FLAT
L=/tmp/crvfs_lookup_$$; mkdir -p $L; chmod 700 $L
trap 'rm -rf $L; echo "deleted $L"' EXIT
trap 'exit 143' TERM INT
for f in LookupTable_3200_0 photonMap.root; do
  $V/bin/python -m mu2efastsim.cvmfsget mu2e.opensciencegrid.org /DataFiles/CRVConditions/v6_0/$f $L/$f || { echo "fetch $f failed"; exit 3; }
done
ls -la $L; export CRVFS_LOOKUP_DIR=$L

echo; echo "==== tile isolation (FLAT, ZE_AFFINITY_MASK=3)"
ZE_AFFINITY_MASK=3 $V/bin/python -c "import jax; print(len(jax.devices()), jax.devices())" 2>&1 | grep -v experimental

echo; echo "==== pytest -rA (tile 0)"
ZE_AFFINITY_MASK=0 $V/bin/python -m pytest -q -rfE -p no:cacheprovider crv/tests/test_reco_jax.py crv/tests/test_photons.py core/tests 2>&1 | tail -12

B="$V/bin/python -m crvfastsim.response.bench --repeat 3"
cores() { local i=$1; [ $i -lt 6 ] && echo "$((1+8*i))-$((8+8*i))" || echo "$((53+8*(i-6)))-$((60+8*(i-6)))"; }
echo; echo "==== one tile (real table 3200_0, real photon map)"
ZE_AFFINITY_MASK=0 taskset -c $(cores 0) $B --table 3200_0 --rows $N --numpy-rows 0 --seed 20261020 --outdir $O/ph_1 > $O/ph_1.txt 2>&1
echo "photons $(tail -1 $O/ph_1.txt)"
ZE_AFFINITY_MASK=0 taskset -c $(cores 0) $B --stage sipm --photon-map $L/photonMap.root --sipms $NS --numpy-sipms 0 --seed 20261120 \
  --outdir $O/si_1 > $O/si_1.txt 2>&1
echo "sipm $(tail -1 $O/si_1.txt)"
echo "photons numpy one core: $(taskset -c 1 $B --table 3200_0 --rows 20000 --numpy-rows 20000 --no-jax 2>&1 | tail -1)"

for st in ph si; do
  echo; echo "==== 12 tiles concurrently: $st"
  t0=$(date +%s.%N)
  for i in $(seq 0 11); do
    if [ $st = ph ]; then a="--table 3200_0 --rows $N --numpy-rows 0 --seed $((20261020+i))"
    else a="--stage sipm --photon-map $L/photonMap.root --sipms $NS --numpy-sipms 0 --seed $((20261120+i))"; fi
    ZE_AFFINITY_MASK=$i taskset -c $(cores $i) $B $a > $O/${st}12_$i.txt 2>&1 & pids+=($!)
  done; wait "${pids[@]}"; pids=()   # explicit pids: a bare wait also waits for the tee of exec > >(tee)
  echo "wall $(awk -v a=$t0 -v b=$(date +%s.%N) "BEGIN{printf \"%.1f\", b-a}") s (includes 12 parallel compiles and table loads)"
  for i in $(seq 0 11); do echo "tile $i $(tail -1 $O/${st}12_$i.txt | cut -c1-260)"; done
  $V/bin/python - $O $st <<'EOF'
import json, sys, glob
o, st = sys.argv[1], sys.argv[2]
key = "jax_rows_per_s" if st == "ph" else "jax_sipms_per_s"
r = []
for f in sorted(glob.glob(f"{o}/{st}12_*.txt")):
    try: r.append(json.loads(open(f).read().strip().splitlines()[-1])[key])
    except Exception: pass
print(f"{st}: {len(r)}/12 tiles ok; sum {sum(r):.0f}/s; per tile min {min(r, default=0):.0f} max {max(r, default=0):.0f}")
EOF
done
echo "outdir $O"; echo exit=0
