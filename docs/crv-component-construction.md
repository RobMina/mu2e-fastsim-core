# Canonical CRV component: bounded construction validation

Status: the 2026-11-06 user build is verified against Offline v13_39_00. `ConstructCaptureCRVModule` passes nominal
and transformed one-event strict audits, all placement overlap checks and runtime inventory/identity checks.
The earlier uniform-box baselines remain preserved. These checks validate bounded construction and coordinate
covariance, not physics-distribution equivalence, a full-detector partition or a production bank.

## Component boundary and ownership

The pinned `Mu2eG4/src/constructCRV.cc` has sector and layer mothers, but **no G4 module mother**. A module consists
of bars and optional CMBs inside module-layer air boxes, absorber sheets, strongback/cover sheets, and its FEB boxes.
The FEBs belong to the module data but are placed directly in hall air, outside the sector mother. Sector-spanning
mothers are air/navigation wrappers, not additional continuous aluminum. Copying a sector mother or scintillator
bounds alone therefore loses module-owned material.

`CosmicRayShield::getSupportStructures()` is separate global geometry with names/materials/positions, not module
ownership. Do not copy a shared support into every module. A component envelope intersecting shared support needs
an explicit, nonduplicating ownership treatment; silently replacing the support with air is not acceptable.
Staggered-layer bounding boxes can overlap neighboring modules even when their material does not. An isolated box
fixture is not automatically a nonoverlapping full-detector partition.

## Minimal construction route

1. Add an opt-in `ConstructCaptureCRVModule` study-world tool. Build the pinned geometry **data** through
   `CosmicRayShieldMaker(config, 0.)`, with `hasCosmicRayShield=true`, and retain the selected shield/module.
   Do not call whole-sector `constructCRV` or expect study-mode GeometryService to install the standard CRV geometry.
2. Reproduce all selected module material boxes from that data: layer mothers with bars/CMBs, absorbers,
   strongback/cover and every `module.getFEBs()` entry. Preserve original bar copy numbers and emit a mapping to
   shield/module/layer/bar; local sensitive identity must not be inferred from a receiving detector-global number.
3. Bound every constituent, including electronics; do not use sector/sheet-only extents. Give the envelope a
   documented clearance and the source hall-air material. Local origin is the computed bounds center, with axes
   parallel to the source Mu2e axes; apply the proper rotation/translation only to the envelope.
4. Use the existing touchable-ancestry recorder with no field or detector SD. Validate constituent/material counts,
   dimensions, containment and overlaps before transport, then local/world transforms and complete trial counts.
5. Compare isolated nominal and transformed placements before designing shared-air/support/neighbor ownership.
   The same construction machinery must accept side orientations, not a top-only approximation.

No production `constructCRV`, maker/data class or transport-recorder behavior needs to change for this geometry
slice. New direct dependencies must be mirrored into CMake and SConscript. The user runs any required build.

## Candidate and explicit choices

The runtime-checked side-module candidate is `Mu2eG4/geom/crv_counters_v10.txt`, `CRV_R1`, module 0. Its inventory
is now measured, but it does not reproduce extracted KPP/v04 or Run 1A/v01 samples. Reuse each sample's actual
pinned construction when comparing to those samples.

The v10 file sets `crs.hideCRVCMBs=true`. A source-faithful comparison must retain that omission; a full-electronics
variant must explicitly set it false and be identified separately. Do not silently enable CMBs and call it identical
geometry. FEB boxes are still module-owned. Global `CRV_Support_T` is a separately configured steel plate and not
per-module material. Select and document the source/CMB variant before generating a module reference.

Study mode skips `ConstructMaterials::constructMu2eMaterials()`. NIST materials alone cannot supply
`ElectronicsCMB`/`ElectronicsFEB`. The required closure is `Polyethylene`, `Electronics` and those two aliases.
The exact definitions are now shared by standard and study construction, preserving recipes and avoiding duplicate
registration rather than copying approximations or enabling all standard materials. Standard calls remain at the
original creation-order positions by static review. Post-build `standard-materials.fcl`, derived from stock
`gdmldump_run1_a`, completed with a full material table confirming Polyethylene 0.956 g/cm3 and
Electronics/FEB/CMB 0.58 g/cm3 with source element fractions. Minimal physics and a constant stacking cut rejecting
all primaries make this debug initialization, not physics capture. There is no pre-extraction runtime A/B; recipe/order review is static
only. This extraction is not part of the retained-input CRV PR.

The pinned `CRSScintillatorShield::getAluminumSheetMaterialName()` returns the absorber-material member. Both names
are `G4_Al` in v10, so that candidate is unaffected; do not silently fix the accessor in this capture work.

## Runtime checkpoint

All placement overlap checks pass. Runtime confirms 4 layers, 64 scintillator bars at 1.06 g/cm3, 6 FEBs at 0.58,
5 aluminum placements (3 absorbers/2 sheets) at 2.699, and 5 air placements at 0.00120479. Including the two vacuum
ancestors gives 82 volume rows; the component itself has 80 placements and 64 mapped bar IDs. The 128 omitted CMBs
are deliberate source geometry, not missing capture rows. Constituent dimensions are archived at full precision;
the shorter source-origin summary must not be treated as full-precision evidence for all source bounds.

Nominal/transformed one-event audits retain 90/86 tracks, 202/199 steps, one exit each and 89/85 natural ends;
component deposits are 36.961493634894936/33.59866941329343 MeV, with 2.470540323626122e-23 MeV approach deposit
each. The nominal local entry is (-111.58850000000007,0,0) mm with p=999.9999999999998 MeV/c along +x at
2.9799172350778145 ns. Primary traversal confirms 888.4115 mm vacuum, then 99.526 mm air to aluminum at
x=-12.0625 mm, then layer/bar (3,9), (2,8), (1,7), (0,6); FEBs are missed.

The transformed fixture uses R=[[.48,-.8,.36],[.64,.6,.48],[-.6,0,.8]] and t=(123,-87,41) mm. Its local entrance
agrees to 8.6e-14 mm and 5.7e-14 MeV/c. This demonstrates one-event coordinate covariance, not physics-distribution
equivalence. An initial duplicate-point SimpleConfig failure is preserved; the successful transformed input uses
standalone explicit lists, not replacement permission. Strict-audit-first vectorized inventory/primary-traversal
reports are archived for both trials; 203 combined local tests pass, including 18 inventory tests.

A fresh one-event box regression also passes (57 tracks/116 steps, 18.541152425084263 MeV component deposit),
without replacing earlier baselines. The dedicated `crv-r1-feb.fcl` input from (-1000,-2035,-63) mm along +x
(run 8, seed base 314659) passes strict one-event audit: 129 tracks/272 steps, one exit/128 natural ends,
32.759215921177955 MeV component deposit. Its primary traverses FEBs 3/0 in 32 steps, 20.00144444840868 mm,
1.2344484228089634 MeV direct primary deposit. Those FEB sums exclude children, which remain in the full capture.
`crv-r1-feb.gdml` confirms ElectronicsFEB density 0.58 g/cm3 and elemental mass fractions Si .103, Al .034,
Au .034, Cu .691, C .118166876291217, H .0198331237087827. This GDML is a constructed-material checkpoint,
not a canonical full physics/runtime manifest; it contains build-local pointer suffixes and referenced materials.

## Remaining limits

Material inventory and boundary mechanics are necessary but do not establish verified physics/material manifests,
complete energy accounting, general boundary-born ownership, a neighboring-module partition, a continuously
conditioned response law, or statistical portability. Broader electronics/passive coverage, dedicated neutral/stopping
transport and independent response-distribution comparisons remain open. `bankAdmissible=false` always; keep experimental capture
distinct from a production bank. No tarball or UVA dispatch follows from these checks.
