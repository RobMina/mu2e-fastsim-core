#!/bin/bash
#PBS -N m2fs_probe
#PBS -A GenesisHackathonOct26
#PBS -q debug
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=00:50:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o m2fs_probe.out
# Aurora step-0 probe for the Mu2e fastsim: node/network inventory, venv with jax[oneapi] on Flare, JAX on the PVC
# tiles (float64), pytest and synthetic CRV benches on one tile and on CPU. Submit: qsub [-v M2FS_FLARE=...] probe_aurora.sh
# Synthetic inputs only: no Mu2e-internal file is read or written.
PROJ=${PBS_ACCOUNT:-GenesisHackathonOct26}
F=${M2FS_FLARE:-/lus/flare/projects/$PROJ/$USER/mu2efastsim}
JAXV=${JAXV:-0.11.2}
O=$F/probe/$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}
mkdir -p "$O" || { echo "cannot create $O (set M2FS_FLARE)"; exit 2; }
exec > >(tee "$O/probe.log") 2>&1
s() { echo; echo "==== $*"; }
s "node"; echo "host=$(hostname) job=${PBS_JOBID:-none} date=$(date -Is) out=$O"
lscpu | grep -E "^(Model name|Socket|Core|Thread|NUMA node\(s\))"; free -g | head -2
s "storage"; df -h /tmp /dev/shm "$F" 2>&1; lsblk -d -o NAME,SIZE,TYPE,MOUNTPOINT 2>&1 | head
ls -d /local/scratch /local /scratch 2>&1
s "containers/fuse"; ls -l /dev/fuse 2>&1; cat /proc/sys/user/max_user_namespaces 2>&1
module use /soft/modulefiles 2>/dev/null; module -t avail apptainer 2>&1 | head -3; ls -d /cvmfs/* 2>&1 | head -3
s "gpus"; command -v xpu-smi >/dev/null && xpu-smi discovery 2>&1 | grep -E "Device Name|Device ID" | head -14
echo "ZE_FLAT_DEVICE_HIERARCHY=${ZE_FLAT_DEVICE_HIERARCHY:-unset} ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-unset}"
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128 HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=$HTTP_PROXY https_proxy=$HTTPS_PROXY no_proxy="localhost,*.alcf.anl.gov"
s "network via proxy (http code, 000 = no route)"
for u in https://github.com https://pypi.org/simple/jax/ \
  http://cvmfs-s1fnal.opensciencegrid.org:8000/cvmfs/mu2e.opensciencegrid.org/.cvmfspublished \
  "https://dbdata0vm.fnal.gov:9443/QE/mu2e/prod/app/SQ/query?dbname=mu2e_conditions_prd&t=val.tables&c=tid&o=tid"; do
  echo "$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$u") $u"
done
s "code"; CODE=$F/code
if [ -d $CODE/.git ]; then git -C $CODE pull -q --ff-only; git -C $CODE submodule update -q --init --recursive
else git clone -q --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git $CODE; fi
git -C $CODE log -1 --format="umbrella %h %ad %s" --date=iso; git -C $CODE submodule status
s "venv"; module load frameworks 2>&1 | tail -1; BASEPY=$(command -v python3); echo "base $BASEPY $($BASEPY --version)"
V=$F/venv; [ -x $V/bin/python ] || $BASEPY -m venv $V
export PIP_CACHE_DIR=/tmp/pipcache_$USER
t0=$SECONDS
$V/bin/python -m pip install -q --upgrade pip
$V/bin/python -m pip install -q "jax[oneapi]==$JAXV" numpy h5py pyarrow pandas awkward uproot matplotlib pytest 2>&1 | tail -3
for p in core crv tracker; do $V/bin/python -m pip install -q --no-deps -e $CODE/$p; done
echo "pip $((SECONDS-t0)) s"; $V/bin/python -m pip freeze | grep -iE "^(jax|jaxlib|jax-oneapi|numpy|intel-sycl|dpcpp)"
module unload frameworks 2>/dev/null
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
s "jax on the tiles"
$V/bin/python - <<'EOF'
import time, jax, jax.numpy as jnp
print("jax", jax.__version__, "backend", jax.default_backend(), "n_devices", len(jax.devices()))
print("devices", jax.devices())
d = jax.devices()[0]; print("device_kind", getattr(d, "device_kind", "?"))
k = jax.random.PRNGKey(0)
for dt in (jnp.float32, jnp.float64):
    x = jax.device_put(jax.random.normal(k, (4096, 4096), dt), d)
    with jax.default_matmul_precision("highest"):
        f = jax.jit(lambda a: a @ a); f(x).block_until_ready()
        t = time.perf_counter(); y = f(x).block_until_ready(); dt_s = time.perf_counter() - t
    ref = (x[:64].astype(jnp.float64) @ x.astype(jnp.float64))
    err = float(jnp.max(jnp.abs(y[:64] - ref)) / jnp.max(jnp.abs(ref)))
    print(f"{jnp.dtype(dt).name}: {2*4096**3/dt_s/1e12:.2f} TFLOP/s, rel err vs f64 {err:.2e}, dtype {y.dtype}")
s = jax.jit(lambda a: jnp.sort(a))(jax.random.uniform(k, (1 << 22,), jnp.float64)); print("sort ok", bool(jnp.all(s[1:] >= s[:-1])))
c = jax.lax.scan(lambda c, v: (c + v, c), 0.0, jnp.arange(1000.0))[0]; print("scan ok", float(c) == 499500.0)
EOF
cd $CODE
s "pytest (tile 0)"; ZE_AFFINITY_MASK=0 $V/bin/python -m pytest -q -p no:cacheprovider crv/tests/test_reco_jax.py core/tests 2>&1 | tail -4
s "reco-bench synthetic, tile 0"
for st in pulses chain; do
  ZE_AFFINITY_MASK=0 timeout 600 $V/bin/python -m crvfastsim reco-bench synthetic --stage $st --max-events 100000 \
    --length 12 --batch 65536 --repeat 2 --outdir $O/xpu_$st > $O/xpu_$st.txt 2>&1; echo "$st exit=$? $(tail -1 $O/xpu_$st.txt)"
done
s "response bench synthetic, tile 0 (photons, sipm)"
ZE_AFFINITY_MASK=0 timeout 400 $V/bin/python -m crvfastsim.response.bench --rows 1000000 --numpy-rows 0 --repeat 2 \
  --outdir $O/xpu_ph > $O/xpu_ph.txt 2>&1; echo "photons exit=$? $(tail -1 $O/xpu_ph.txt)"
ZE_AFFINITY_MASK=0 timeout 400 $V/bin/python -m crvfastsim.response.bench --stage sipm --sipms 200000 --numpy-sipms 0 \
  --repeat 2 --outdir $O/xpu_si > $O/xpu_si.txt 2>&1; echo "sipm exit=$? $(tail -1 $O/xpu_si.txt)"
s "CPU (one process, numpy + JAX cpu)"
JAX_PLATFORMS=cpu timeout 300 $V/bin/python -m crvfastsim.response.bench --rows 20000 --numpy-rows 20000 --no-jax \
  > $O/cpu_ph.txt 2>&1; echo "photons numpy exit=$? $(tail -1 $O/cpu_ph.txt)"
JAX_PLATFORMS=cpu timeout 300 $V/bin/python -m crvfastsim.response.bench --stage sipm --sipms 20000 --numpy-sipms 20000 \
  --no-jax > $O/cpu_si.txt 2>&1; echo "sipm numpy exit=$? $(tail -1 $O/cpu_si.txt)"
JAX_PLATFORMS=cpu timeout 300 $V/bin/python -m crvfastsim reco-bench synthetic --stage chain --max-events 20000 \
  --length 12 --repeat 2 --outdir $O/cpu_chain > $O/cpu_chain.txt 2>&1; echo "chain jax-cpu exit=$? $(tail -1 $O/cpu_chain.txt)"
s "done"; echo "outdir $O"; echo exit=0
