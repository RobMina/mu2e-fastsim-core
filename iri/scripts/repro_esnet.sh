#!/bin/bash
# ESnet pod: R5/P6 reco-repro (synthetic) with a small event count, so the npz stays under the 2 MiB IRI download cap.
NEV=${NEV:-1000}; export JAX_PLATFORMS=cpu; cd /opt/mu2e-fastsim
grep -m1 'model name' /proc/cpuinfo
python -m crvfastsim reco-repro synthetic --max-events $NEV --outdir /tmp/r_$$ 2>&1 | tail -2
cp /tmp/r_$$/* "$OLDPWD"/ && rm -rf /tmp/r_$$; ls -la "$OLDPWD"; echo exit=0
