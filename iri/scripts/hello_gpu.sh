#!/bin/bash
# Week-1 Polaris smoke test: node and GPU inventory, then a JAX matmul on every A100, from the project venv.
VENV=${M2FS_EAGLE:?}/venv
echo "host=$(hostname) date=$(date -Is) pbs_job=${PBS_JOBID:-none}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "python=$($VENV/bin/python --version 2>&1)"
$VENV/bin/python - <<'EOF'
import time
import jax, jax.numpy as jnp
print("jax", jax.__version__, "devices", jax.devices())
for d in jax.devices():
    x = jax.device_put(jnp.ones((8192, 8192), jnp.float32), d)
    (x @ x).block_until_ready()
    t = time.perf_counter(); y = (x @ x).block_until_ready(); dt = time.perf_counter() - t
    print(f"{d}: 8192^3 fp32 matmul {dt*1e3:.1f} ms = {2*8192**3/dt/1e12:.1f} TFLOP/s, y[0,0]={float(y[0,0])}")
EOF
echo "exit=$?"
