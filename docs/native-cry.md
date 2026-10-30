# Standalone CRY backend and container extension

CRY 1.7 is a host-CPU correlated cosmic-shower generator. This opt-in backend
uses its published C++ implementation through an original C ABI and Python
ctypes adapter. It does not depend on art, Geant4, ROOT, CLHEP, CVMFS, Spack or a
GPU. `cry_native.CrySource` exposes raw CRY coordinates and must not be fed
directly to Mu2e transport. `cosmic_cry.CosmicSource` adds the extracted-v04
stock-style coordinate map, five-face line projection, energy cuts, accepted
shower grouping, gen-table schema and CosmicTimeOffset convention. This is an
opt-in reference implementation, not an independently certified production default.

## User-run Apptainer build

`native/cry/cry-overlay.def` extends an existing fastsim SIF without modifying it.
Run from the core checkout root on an allocated CPU node with Apptainer, user
namespaces/fakeroot, and outbound HTTPS to Debian package mirrors and LLNL.
No privileged host install is needed. It compiles in a temporary build stage;
only the stripped shared shim, tables, notices, checksums, Python adapter and
tests enter the final stage. Inputs copied by `%files` are staged under
`/opt/cry-build`, not `/tmp`: Apptainer can bind a separate temporary directory
over the image's `/tmp` during `%post`. Input-readability checks run before any
package installation. A C++ compiler is not added to the runtime stage.
The existing installed `mu2efastsim` package receives the new module only;
other fastsim code and the base image remain untouched.

```bash
# First obtain an interactive CPU allocation at your Slurm site.
module load apptainer
cd /path/to/mu2e-fastsim-core
export APPTAINER_TMPDIR=$(mktemp -d /path/to/own/scratch/cry-build.XXXXXX)
export APPTAINER_CACHEDIR="$APPTAINER_TMPDIR/cache"
mkdir -p "$APPTAINER_CACHEDIR"
bash native/cry/prepare-extraction.sh "$APPTAINER_TMPDIR/extraction"
bash native/cry/check-extraction.sh "$APPTAINER_TMPDIR/extraction"
export APPTAINER_CONFIG_FILE="$APPTAINER_TMPDIR/extraction/apptainer.conf"
apptainer build --fakeroot \
  --build-arg BASE_IMAGE=/path/to/mu2efastsim.sif \
  --mksquashfs-args "-mem 4G -processors 4" \
  /path/to/new-mu2efastsim-cry.sif native/cry/cry-overlay.def
```

Do not reuse the output path for an unrelated image. Packing memory/processes
are bounded explicitly; request enough RAM for the image build. Extraction is
separate: Apptainer 1.4.5 supplies no unsquashfs argument flag and gives its
bundled extractor precedence over a normal PATH wrapper. The preparation script
copies the site's config, changes only `binary path` to put a private wrapper
first, and limits extraction to four processors with 64 MiB queues. The wrapper
also implements Apptainer's `_WRAPPER_EXEC_CMD` probes: `echo` returns only the
underlying ELF path; the dynamic-loader probe gets the ELF path with no extractor
flags. Without this protocol Apptainer fails before extraction. A custom
config requires a non-setuid installation (as on the target cluster); do not
attempt to bypass site restrictions on a setuid installation. The helper runs
no build and installs nothing on the host. `check-extraction.sh` exercises the
wrapper in a minimal contained filesystem using the same binary/library-bind
pattern as Apptainer; it runs `unsquashfs -version`, not a build or full image
extraction. It requires host `readelf` (binutils). Scratch is used for temporary roots
because `/tmp` may be tmpfs and charge unpacked image pages to the job's RAM.
This recipe is for the existing Debian Bookworm fastsim base, not an arbitrary OS image.
The build's `%test` checks table hashes, binding contracts, seed repeatability,
complete-shower batch boundaries and the cumulative live-time clock. These are
software smoke checks, not a physics-validation certificate. If a pinned URL or
checksum changes, stop and inspect the artefact; do not remove the checksum gate.

After a successful build:

```bash
apptainer test /path/to/new-mu2efastsim-cry.sif
apptainer exec --cleanenv /path/to/new-mu2efastsim-cry.sif \
  python3 -m mu2efastsim.cry_native --showers 100 --seed 1
apptainer exec --cleanenv /path/to/new-mu2efastsim-cry.sif \
  cat /opt/cry/VERSION /opt/cry/BUILD.sha256
```

The user runs builds. Deployment of the extended image as a production default
must wait for stock-generator and end-to-end validation. Existing workflows keep
using their existing image and table-driven generator until then.

## Mu2e wrapper from a code snapshot

The image build above contains the raw backend; a later Python wrapper can be
used without rebuilding C++ by bind-mounting this core checkout and setting
`PYTHONPATH` to it. For an extracted-v04 GDML matching the stock sample:

```python
from mu2efastsim.cosmic_cry import CosmicConfig, CosmicSource
config = CosmicConfig.extracted("mu2e_extracted_v04.gdml")
with CosmicSource(config, seed=1234) as source:
    gen, subruns, receipt = source.generate(50000, subrun=1)
```

The wrapper uses reference (-3904,20000,24171) mm and the extracted target box
from Production's extracted CRY configuration. It reads centered world and
HallAir bounds from GDML. The factory admits only the validated GDML SHA256
`5b357107ac740de5df770066b6428c1a58ec61e0349380219ab08dd9d1ace3d9`;
unknown or named-run GDMLs fail closed rather than inheriting a guessed reference
point. An explicit `CosmicConfig` is required for separately verified geometries.
Only the extracted configuration is admitted by the chain hook.
Particle masses follow Offline's `ParticleList.txt`. CRY is configured for all
particle types, 300 m subbox, latitude 41.8, altitude 0 and 2021-06-21.

Per-particle KE <50 MeV is removed. The upper shower cut is 200 GeV of **total**
energy after that cut, including particles missing the target. Intersections
use full lines, inclusive edges, roof and four sides (no floor); selected
particles are placed at the highest-y **world** intersection, not on the target.
Raw times reproduce `secondary.t - timeSimulated()` numerically (no flight-time
shift and no seconds-to-ns multiplication). One independent uniform offset on
[0,100005] is rounded to float32 and added to all particles of each accepted
event, matching `CosmicTimeOffset`'s convention. The random sequence is not
CLHEP/art bit-identical.

The stock over-energy rejection loop retains projected particles from an
over-energy shower until a subsequent under-cap shower completes an event.
This implementation reproduces that behaviour explicitly, including carries
across raw-batch boundaries; it does not silently fix stock. Raw CRY shower
identity and accepted art-event identity therefore differ in those cases.
Receipts report over-cap raw draws and carried rows, including any prefetched
batch beyond the final exposed event. The exposed live time stops at the last
requested accepted event; buffered events are retained for the next call.

Use `python3 -m mu2efastsim.cry_validate --help` for independent sample comparisons.
Errors are sample standard errors over independent shards, not paired errors;
K=1 has indeterminate errors. Binned spectra include target and world positions.
Rare particle species may remain statistically untested even when the bulk rate
agrees. No blanket physics-closure claim follows from a smoke test.

## Inputs, dependencies and licensing

The source is fetched from
<https://nuclear.llnl.gov/simulation/cry_v1.7.tar.gz>, pinned to SHA256
`dcee2428f81cba113f82e0c7c42f4d85bff4b8530e5ab5c82c059bed3e570c20`.
The build compiles `src/CRY*.cc`, excluding the unused Fortran wrapper, with
`g++ -std=c++17 -O2 -fPIC -shared`; the shim uses a seeded C++ standard RNG.
This preserves CRY sampling, not art/CLHEP random-number sequence identity.
Runtime dependencies are libstdc++, libgcc, libc and libm. Python and numpy
are already in the base image; ctypes is in the Python standard library.

All three altitude tables (9,767,111 bytes total) are bundled and checksum-verified.
Although the extracted setup uses sea level, CRYSetup 1.7 eagerly opens **all
three** altitude files on initialisation, so do not install only the sea-level
file without changing upstream behaviour.

CRY is not Apache-2.0 code. The upstream `src/COPYRIGHT.TXT` gives a three-clause
BSD-style redistribution notice, copyright 2007–2012 Regents of the University
of California, LLNL UCRL-CODE-227323. The recipe preserves it and the additional
LLNL/DOE notice linked from it:
<https://nuclear.llnl.gov/simulations/additional_bsd.html>. Both are in
`/opt/cry/share/licenses/`; the runtime label records the third-party licence.
The original adapter/shim remain under this repository's Apache-2.0 licence.
No CRY source or tables are vendored into this repository.

## Raw API and limits

```python
from mu2efastsim.cry_native import CrySource
with CrySource(seed=1234) as source:
    batch = source.next_showers(100)
    rows = batch.particles
    seconds = batch.live_seconds
```

Lengths are metres, energies MeV, times seconds, and axes are CRY's own axes.
The `shower` index is local to each batch. Empty showers are retained in the
per-shower clock, not represented as fake particles. All particles are returned
without a caller-array truncation cap; generate bounded batches to limit RAM.
CRY's own upstream generation settings and rejection rules still apply.
Each source owns an independently seeded RNG and cached batch buffers; Python
copies the buffers before the next call. Calls are serialised within the shared
library, with a per-source Python lock protecting generate/copy/close.

CRY 1.7 does not implement complete destruction of its internal allocations.
Use one long-lived source per worker; process exit is the reliable reclamation
boundary. This shim does not claim to fix upstream memory ownership. C++
exceptions are translated to Python errors, but upstream asserts/abort cannot be
caught this way. Use verified bundled tables and validated settings, not
untrusted CRY table files. No thread-throughput or memory-growth certification
has been performed yet.

## Native `chain_point` admission

The evaluator supports extracted-only `gen_set=4` in addition to its unchanged
0–3 table sets. An example request (not a submission command):

```json
{"task":"chain_point", "shards":2, "files_per_job":1, "shard_offset":0,
 "max_events":0, "seed":1234, "wall_minutes":60, "gen_set":4,
 "gen_events":50000, "run_config":0, "chain_flags":0, "tier1b":0}
```

`gen_events` is accepted events per sub-job, 1..2**20−1. It defaults to zero and
is removed from normalized table requests, preserving their persisted identity.
Native mode requires files_per_job=1, shard_offset=max_events=0, run_config=0
and no prefetch bit. The ordinary runchain event-count/key limits also apply.
Each sub-job owns a fresh source with generator seed `seed+1000*k`, subrun `k+1`,
run 1400 and events 1..gen_events. Transport uses its own RNG with the existing
per-sub-job seed convention. Across different points the caller must select
nonoverlapping generator seeds; no automatic global replica registry is implied.

No gen-file fingerprints, transfer streams, cache entries or wait-budget bytes
are created for set 4. Geometry, conditions and tier-1b inputs still use the
existing pinned-manifest, verified-ready-cache, job-consumption-guard and tmpfs
cleanup contracts. The extracted GDML signature is checked before code
preparation and again by the native wrapper. Code and the CRY-enabled SIF are
pinned by the normal execution manifest; the bundled CRY tables/shim are covered
by that image fingerprint. Lookup/include closure retains the documented gaps
in [artifact-bindings.md](artifact-bindings.md). A job probes CRY before requesting
streamed inputs. A successful `check` is parameter/resource admission, not remote
runtime verification or physics certification.

The wall-budget heuristic adds 900 s/million native accepted events to the
existing 180 s/million chain allowance plus 60 s setup, and retains the input-wait
budget and SU cap. This is a conservative admission allowance, not a calibrated
runtime prediction; it must not be used as a measured throughput claim.
Generation, including source initialization and Python wrapper processing, is
measured separately by `timers.generation_s` and included in chain wall time.

Native `--emit` records retain exposure and the full per-source receipt. Merge
refuses mixed native/table inputs, changed generator configurations, duplicate
fresh-source seeds and overlapping event ranges. Chain-point results additionally
verify the expected count, seed, geometry signature and identifiers against the
bound request. Native result metadata holds the summed live exposure, generation
seconds, receipts and clusters/live-second. Existing endpoint metrics and table
results keep their previous definitions; no paired-event comparison is implied.

Use a matching public committed core/crv snapshot and explicitly select the
CRY-enabled SIF before live deployment. Dirty worktrees are deliberately not
accepted by the endpoint snapshot contract. No new image build is needed when
only the bind-mounted Python code changes. Unit/fixture validation and a local
emitted-shard smoke do not certify live SSH/Slurm streaming or GPU throughput.

The merge event-overlap guard uses the packed subrun/event identity, not the run
number: changing run alone cannot disambiguate the chain's packed labels.

## On-device generation

The CPU binding above is the reference, not an on-device generator. The
experimental JAX port draws CRY showers and performs Mu2e acceptance/projection
on the GPU: [device-cry.md](device-cry.md). Select `runchain --gen device` or
`chain_point gen_set=5`; set 4 remains C++ CPU. Only PDF preparation/upload and
the selected-primary DataFrame boundary remain on the host. No CPU-generated
showers are uploaded as a substitute for device sampling. See that document for
bounded-pool failures, RNG/shape semantics, validation and cold/warm timing scope.
