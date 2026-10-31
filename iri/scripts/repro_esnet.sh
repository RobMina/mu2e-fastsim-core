#!/bin/bash
# ESnet pod: R5/P6 reco-repro (synthetic) with a small event count, so the npz stays under the 2 MiB IRI download cap.
NEV=${NEV:-1000}; export JAX_PLATFORMS=cpu; T=$(mktemp -d); cd /opt/mu2e-fastsim
grep -m1 'model name' /proc/cpuinfo
python -m crvfastsim reco-repro synthetic --max-events $NEV --outdir $T/r 2>&1 | tail -2
cp $T/r/* "$OLDPWD"/ && rm -rf $T; ls -la "$OLDPWD"; echo exit=0
