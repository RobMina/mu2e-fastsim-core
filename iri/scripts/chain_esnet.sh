#!/bin/bash
# ESnet pod: synthetic CRV reco chain bench (pulses -> coincidences), JAX on CPU and the numpy reference; outputs in /tmp.
NEV=${NEV:-20000}; export JAX_PLATFORMS=cpu; O=$PWD; cd /opt/mu2e-fastsim
echo "host $(hostname) job ${JOB_ID:-none} cpus $(nproc) quota $(cat /sys/fs/cgroup/cpu.max)"; grep -m1 'model name' /proc/cpuinfo
for s in chain pulses coinc; do
  echo "==== reco-bench synthetic --stage $s ($NEV events)"
  python -m crvfastsim reco-bench synthetic --stage $s --max-events $NEV --outdir /tmp/rb_$s 2>&1 | tail -2 | cut -c1-1500
  cp /tmp/rb_$s/*.json $O/ 2>/dev/null
done
rm -rf /tmp/rb_*; ls -la $O; echo exit=0
