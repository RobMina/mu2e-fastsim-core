# CRY on the device (experimental, opt-in)

`mu2efastsim.cry_device` re-implements CRY 1.7's published correlated sampler
with JAX float64 kernels. It does **not** upload CPU-generated showers and does
not call the native C++ sampler. The CPU binding remains the independent
reference. CRY distribution tables are read once on the host, hash-verified,
converted to CDFs and uploaded to the selected device. No events are generated
on the host in that setup. See [third-party/cry.txt](third-party/cry.txt) for the
retained LLNL/DOE notices; distribution tables are not vendored.

## What executes where

Host setup reads only the pinned `cosmics_0.data` (default `/opt/cry/data`, or
`CRYDATAPATH`). It reproduces CRYPrimary's solar/latitude cutoff, 1000-point
left-Riemann partial-rate integral, 10k-bin primary energy PDF and shower
weighting. Allowed settings are currently sea level, 300 m square, latitude
41.8, 2021-06-21, all seven species. Unknown table hashes fail closed.

JAX kernels draw **primary energy and live time, multiplicity, all particle
species, kinetic energies, times, charges, angles, azimuths and lateral
positions on the GPU**. A common primary energy bin conditions every secondary
in a shower. This preserves CRY's actual correlation model, not independent
muon or marginal-spectrum draws. CRY's three-valued charge distribution and
signed PDG convention, logarithmic/linear within-bin sampling and discrete
multiplicity support are retained. Zero-mass conditional PDF rows have safe
padding for computation but an active draw from one is a fatal model error,
never a fallback spectrum.

Mu2e map, KE cut, full-shower total-energy cap, five-face full-line target and
world intersections, highest-y world position, overenergy carry grouping,
accepted clocks and shared float32 event offsets also run on the GPU. The
pending carry buffer remains device-resident between batches. Full raw rejected
showers are never copied to the host by the ordinary accepted-event source.
Only bounded selected-primary buffers and scalar diagnostics cross back to the
existing host DataFrame/NumPy CRV transport boundary. This implements **device
generation**, not an end-to-end device CRV chain. Downstream transport has its
own backend contracts.

## API and chain hooks

```python
from mu2efastsim.cosmic_cry import CosmicConfig
from mu2efastsim.cry_device import DeviceCosmicSource
with DeviceCosmicSource(CosmicConfig.extracted("mu2e_extracted_v04.gdml"), seed=123) as source:
    gen, subruns, receipt = source.generate(50000, subrun=1)
```

The public source requires JAX's GPU backend by default. `require_gpu=False`
exists for CPU kernel-contract tests, not for a production fallback. It will
not silently accept CPU when requested from the CLI. Use the same code snapshot
and CRY-enabled image as the native reference; no C++ rebuild is needed for this
Python/JAX implementation. Only the pinned sea-level table is required by the
device path; the C++ backend independently needs all three altitude tables.

```bash
python3 -m crvfastsim runchain native --gen device --gen-events 50000 \
  --gen-seed 123 --gen-subrun 1 --seed 456 --jax \
  --gdml /inputs/mu2e_extracted_v04.gdml --calib /inputs/crvcalib.csv \
  --deviation /inputs/crvphoton.csv --tier1b strict_v3 \
  --tier1b-dir /inputs/tier1b_strict_v2 --emit /outputs/shard.json --outdir /outputs/chain
```

`chain_point gen_set=5` selects device CRY, with the same extracted-only
`gen_events`, single stream per sub-job, seed/subrun, provenance/cleanup and
unsupported-mode admission as CPU native set 4. It probes GPU/table support
before asking for streamed inputs. Sets 0–3 remain table-driven; set 4 remains
C++ CPU. A live evaluator still needs matching clean public code pins, the
appropriate image and input accessibility; a render test is not deployment.
Device and C++ receipts cannot be merged together as a single generator arm.

## Bounded shapes and RNG

Default raw batch 100,000 showers, flat pool 400,000 particles, selected buffer
8,192 rows. Multiplicities are sampled without a per-shower truncation. The
sum of complete-shower counts sizes the flat pool; exceeding it (or a selected/
pending output pool) makes the entire batch fail before publication. No
clipping, dropping, auto-resampling, adaptive rejection or silent fallback is
allowed. Change capacities explicitly and validate the new configuration;
changing shape can change JAX's RNG stream. Overflow-conditioned successes must
not be pooled while ignoring failed batches. Receipt records capacities/batch
size. The table support still includes rare 1000-particle raw showers.

PRNG is explicit `threefry2x32`; all 64 seed bits enter the key, then batch
index and quantity identify independent draw streams. Reproducible for fixed
seed/shapes/runtime; not bit-identical to art/CLHEP, the C++ shim, or another
batch shape. No cross-runtime bitwise promise is made. Seeds must be selected
independently across replicas and disjoint event identifiers assigned by callers.

## Validation and timing boundaries

`tests/test_cry_device.py` covers PDF semantics, pinned table/rate, raw units/
repeatability/bounded overflow, stock projection and overenergy carry across
batch boundaries. `cry_validate --backend device` compares independent accepted
samples against stock tables with shard-SEs, multiplicity (including multimuon),
signed-PDG rates and target/world/energy/angle/time spectra. The CPU and stock
paths are references, not paired RNG samples. Small-N or absent rare species do
not certify tails.

`python3 -m mu2efastsim.cry_device_check --gdml ... --out ...` requires a GPU,
measures synchronized raw and accepted kernels and separate selected-buffer
copy time; cold compilation is not called steady-state generation. Receipt
`device_setup_s` measures table prep/upload, `device_kernel_wall_s` includes
synchronized sampling/projection/selection (first pass includes compilation),
`device_copy_wall_s` measures the host copy, and the chain's `generation_s`
includes source init, kernels, copies and DataFrame packing. Independent
benchmark results must state hardware, shapes, warm/cold scope and full-chain
boundary; no speed-up is implied merely by using a GPU.

## Measured implementation checks (A100, 2026-10-30)

Ten independent 50k-event device shards versus ten stock 50k-event shards give
accepted rate ratio **0.999780 ± 0.001599** (independent shard standard errors).
Multiplicity mean 1.005982 ± 0.000132 versus 1.005606 ± 0.000117;
multiprimary fraction 0.005730 ± 0.000123 versus 0.005396 ± 0.000115;
multimuon fraction 0.003466 ± 0.000076 versus 0.003200 ± 0.000085. These latter
quantities are about two SE higher. Signed-muon-positive rate is 2.8 SE lower in
this particular device/stock sample. No kaons observed: not validated.

The 2,263 populated spectrum bins (including multiplicity, signed-PDG energy/
zenith/azimuth/position/time and unordered pair xz-distance diagnostics) have no
Bonferroni-family flag. This is descriptive sensitivity, **not equivalence**.
World-plane pair-distance p90 is 44.93 ± 1.42 m versus 41.23 ± 0.57 m (~2.4 SE
high); target-plane p90 is 14.65 ± 0.14 m versus 14.58 ± 0.10 m. Pair-weighted
unordered xz separation is not the event bounding extent or an assumed match
to older differently defined lateral metrics. Against the independently saved
C++ reference, accepted rate ratio is 0.998928 ± 0.001754 and multiprimary and
multimuon ratio excursions are each below two SE. No reference was re-generated.

Warm accepted generation on one A100 takes **0.1727/0.1720 s per 10k accepted
events** (17.27/17.20 us/event) including selected-buffer copies and host packing;
synchronized kernels take 0.1306/0.1299 s. The first cold 10k pass takes 54.32 s
with compilation. Raw 100k showers warm 0.0041/0.0031 s; the C++ diagnostic draw
is 0.1355 s. These few repeated batches are measured implementation timings,
not projected campaign runtimes or a general hardware throughput guarantee.

A 50k device-generated full chain with GPU response/reco, against a saved
independent table-chain arm, gives cluster/live-second ratios EX
1.0319 ± 0.0319, T1 0.9979 ± 0.0719, T2 0.9702 ± 0.0605, M 1.0080 ± 0.0997.
Errors cluster counts within event and are conditional on measured exposure;
no exposure or library/systematic uncertainty is included. Cold full-chain wall
191.26 s includes 54.72 s setup and 56.90 s generation. Separate 5k+5k CLI
processes verified device receipts and merge, not warm worker throughput.
