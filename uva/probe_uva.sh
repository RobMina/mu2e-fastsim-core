#!/bin/bash
#SBATCH -J m2fs_probe
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
# UVA Rivanna/Afton step-0 probe for the Mu2e fastsim: node/storage/network inventory, venv with jax[cuda12] on
# the cluster scratch, JAX on one A100 (float64), pytest and synthetic CRV benches on GPU and CPU. Synthetic inputs only.
# Submit from the laptop (on UVA network/VPN), with F the cluster work area:
#   ssh "$FASTSIMKIT_SLURM_HOST" "mkdir -p $F/probe && sbatch -A $FASTSIMKIT_SLURM_ACCOUNT -o $F/probe/slurm-%j.out --export=ALL,FASTSIMKIT_SCRATCH=$F" < probe_uva.sh
# Site settings come from the environment (site.env.example): FASTSIMKIT_SCRATCH (work area on the cluster), the
# account at submission (sbatch -A "$FASTSIMKIT_SLURM_ACCOUNT"), and the log path (sbatch -o "$FASTSIMKIT_SCRATCH/probe/slurm-%j.out").
F=${M2FS_UVA:-${FASTSIMKIT_SCRATCH:?set FASTSIMKIT_SCRATCH (work area on the cluster scratch; sbatch --export=ALL,FASTSIMKIT_SCRATCH=...)}}
JAXV=${JAXV:-0.11.2}
O=$F/probe/$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-none}
mkdir -p "$O" || { echo "cannot create $O"; exit 2; }
exec > >(tee "$O/probe.log") 2>&1
s() { echo; echo "==== $*"; }
s "node"; echo "host=$(hostname) job=${SLURM_JOB_ID:-none} account=${SLURM_JOB_ACCOUNT:-?} date=$(date -Is) out=$O"
lscpu | grep -E "^(Model name|Socket|Core|Thread|NUMA node\(s\))"; free -g | head -2
echo "cpus_on_node=${SLURM_CPUS_ON_NODE:-?} nproc=$(nproc) TMPDIR=${TMPDIR:-unset}"
s "storage"; df -hT /tmp /dev/shm "${TMPDIR:-/tmp}" "$F" "$HOME" 2>&1 | sort -u
ls -d /local* "$(dirname "$F")/local" 2>&1
t=${TMPDIR:-/tmp}/m2fs_dd_$$; dd if=/dev/zero of=$t bs=1M count=1024 conv=fdatasync 2>&1 | tail -1; rm -f $t
s "containers/fuse"; ls -l /dev/fuse 2>&1; cat /proc/sys/user/max_user_namespaces 2>&1; ls -d /cvmfs/* 2>&1 | head -3
module load apptainer >/dev/null 2>&1; apptainer --version 2>&1   # no pipe: a subshell would drop the module
s "gpu"; nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv 2>&1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
s "network from the compute node (http code, 000 = no route)"
for u in https://github.com https://pypi.org/simple/jax/ https://raw.githubusercontent.com/Mu2e/Offline/main/README.md \
  http://cvmfs-s1fnal.opensciencegrid.org:8000/cvmfs/mu2e.opensciencegrid.org/.cvmfspublished \
  "https://dbdata0vm.fnal.gov:9443/QE/mu2e/prod/app/SQ/query?dbname=mu2e_conditions_prd&t=val.tables&c=tid&o=tid"; do
  echo "$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$u") $u"
done
s "code"; CODE=$F/code
if [ -d $CODE/.git ]; then git -C $CODE pull -q --ff-only; git -C $CODE submodule update -q --init --recursive
else git clone -q --recurse-submodules https://github.com/RobMina/mu2e-fastsim.git $CODE; fi
git -C $CODE log -1 --format="umbrella %h %ad %s" --date=iso; git -C $CODE submodule status
s "venv"; module load miniforge/26.7.2 >/dev/null 2>&1
BASEPY=$(command -v python3); echo "base $BASEPY $($BASEPY --version 2>&1)"
V=$F/venv
$V/bin/python -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null || { rm -rf $V; $BASEPY -m venv $V; }
module unload miniforge 2>/dev/null
export PIP_CACHE_DIR=$(mktemp -d)
t0=$SECONDS
$V/bin/python -m pip install -q --retries 10 --upgrade pip
$V/bin/python -m pip install -q --retries 10 "jax[cuda12]==$JAXV" numpy h5py pyarrow pandas awkward uproot matplotlib pytest 2>&1 | tail -3
for p in core crv tracker; do $V/bin/python -m pip install -q --no-deps -e $CODE/$p; done
echo "pip $((SECONDS-t0)) s"; $V/bin/python -m pip freeze | grep -iE "^(jax|jaxlib|jax-cuda|numpy|nvidia-cuda-runtime)"
sh $CODE/container/fetch_offline_data.sh $F/offline_data && export MU2E_SEARCH_PATH=$F/offline_data
export JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
s "jax on the A100"
$V/bin/python - <<'EOF'
import time, jax, jax.numpy as jnp
print("jax", jax.__version__, "backend", jax.default_backend(), "devices", jax.devices())
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
EOF
cd $CODE
s "pytest (GPU)"; $V/bin/python -m pytest -q -p no:cacheprovider crv/tests/test_reco_jax.py core/tests 2>&1 | tail -4
s "reco-bench synthetic, GPU"
for st in pulses chain; do
  timeout 600 $V/bin/python -m crvfastsim reco-bench synthetic --stage $st --max-events 200000 \
    --length 12 --batch 65536 --repeat 3 --outdir $O/gpu_$st > $O/gpu_$st.txt 2>&1; echo "$st exit=$? $(tail -1 $O/gpu_$st.txt)"
done
s "response bench synthetic, GPU (photons, sipm)"
timeout 400 $V/bin/python -m crvfastsim.response.bench --rows 1000000 --numpy-rows 0 --repeat 2 \
  --outdir $O/gpu_ph > $O/gpu_ph.txt 2>&1; echo "photons exit=$? $(tail -1 $O/gpu_ph.txt)"
timeout 400 $V/bin/python -m crvfastsim.response.bench --stage sipm --sipms 200000 --numpy-sipms 0 \
  --repeat 2 --outdir $O/gpu_si > $O/gpu_si.txt 2>&1; echo "sipm exit=$? $(tail -1 $O/gpu_si.txt)"
s "R5 repro: GPU vs CPU"
$V/bin/python -m crvfastsim reco-repro synthetic --max-events 5000 --outdir $O/repro_gpu > $O/repro_gpu.txt 2>&1
JAX_PLATFORMS=cpu $V/bin/python -m crvfastsim reco-repro synthetic --max-events 5000 --outdir $O/repro_cpu \
  --against $O/repro_gpu/repro_gpu.npz > $O/repro_cpu.txt 2>&1
for r in gpu cpu; do grep -h -oE '"(backend|pass_bit_identical|pass_statistical)": [^,}]*' $O/repro_$r/repro_stats.json 2>/dev/null | tr '\n' ' '; echo; done
s "CPU (JAX cpu, ${SLURM_CPUS_ON_NODE:-?} cores)"
JAX_PLATFORMS=cpu timeout 900 $V/bin/python -m crvfastsim reco-bench synthetic --stage chain --max-events 20000 \
  --length 12 --repeat 2 --outdir $O/cpu_chain > $O/cpu_chain.txt 2>&1; echo "chain jax-cpu exit=$? $(tail -1 $O/cpu_chain.txt)"
s "done"; echo "outdir $O elapsed $SECONDS s"
echo "SU check after the job ends: sacct -j ${SLURM_JOB_ID} -X -o JobID,Elapsed,AllocTRES%60,State; allocations -a ${SLURM_JOB_ACCOUNT:-<account>}"
echo exit=0
