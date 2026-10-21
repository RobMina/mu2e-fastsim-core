#!/bin/bash
# ESnet GPU pod (gpu image): JAX device check, matmul fp32/fp64, CRV response bench with the real v6_0 tables
# (pod /tmp, deleted on exit), synthetic reco chain, R5/P6 repro (npz < 2 MiB). Outputs in /tmp, small JSON to cwd.
O=$PWD; NROWS=${NROWS:-1000000}; NSIPMS=${NSIPMS:-200000}; NEV=${NEV:-20000}
echo "== start $(date -u +%FT%TZ) host $(hostname) job ${JOB_ID:-none}"; cat /opt/mu2e-fastsim/VERSION
echo "cpus $(nproc) quota $(cat /sys/fs/cgroup/cpu.max 2>/dev/null) mem $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
env | grep -E '_VISIBLE_DEVICES|^NVIDIA' | sort
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv 2>&1 || ls -l /dev/nvidia*
L=/tmp/crvfs_lookup_$$; mkdir -p $L; chmod 700 $L
trap 'rm -rf $L; echo "deleted $L ($(ls $L 2>/dev/null | wc -l) files left)"' EXIT
trap 'exit 143' TERM INT
cd /opt/mu2e-fastsim
python - <<'PY'
import time, jax, jax.numpy as jnp
print("jax", jax.__version__, jax.default_backend(), jax.devices())
for dt in (jnp.float32, jnp.float64):
    n = 4096; a = jax.random.normal(jax.random.PRNGKey(0), (n, n), dt); f = jax.jit(lambda x: jnp.matmul(x, x, precision="highest"))
    f(a).block_until_ready(); t = time.time(); r = 5
    for _ in range(r): f(a).block_until_ready()
    print(f"matmul {jnp.dtype(dt).name} {2*n**3*r/(time.time()-t)/1e12:.2f} TFLOP/s")
PY
[ "$(python -c 'import jax;print(jax.default_backend())')" = gpu ] || { echo "JAX not on GPU: stop"; exit 4; }
for f in LookupTable_3200_0 photonMap.root; do
  python -m mu2efastsim.cvmfsget mu2e.opensciencegrid.org /DataFiles/CRVConditions/v6_0/$f $L/$f || { echo "fetch $f failed"; exit 3; }
done; export CRVFS_LOOKUP_DIR=$L
B="python -m crvfastsim.response.bench --repeat 3"
echo "==== photons, real 3200_0 ($NROWS rows)"; $B --table 3200_0 --rows $NROWS --numpy-rows 0 --seed 20261020 2>&1 | tail -1
echo "==== SiPM, real photon map ($NSIPMS SiPMs)"; $B --stage sipm --photon-map $L/photonMap.root --sipms $NSIPMS --numpy-sipms 0 --seed 20261120 2>&1 | tail -1
for s in chain pulses coinc; do
  echo "==== reco-bench synthetic --stage $s ($NEV events)"
  python -m crvfastsim reco-bench synthetic --stage $s --max-events $NEV --outdir /tmp/rb_$s 2>&1 | tail -1; cp /tmp/rb_$s/*.json $O/ 2>/dev/null
done
echo "==== R5/P6 reco-repro (1000 events)"
python -m crvfastsim reco-repro synthetic --max-events 1000 --outdir /tmp/rp 2>&1 | tail -1; cp /tmp/rp/* $O/
rm -rf /tmp/rb_* /tmp/rp; ls -la $O; echo "== done $(date -u +%FT%TZ)"; echo exit=0
