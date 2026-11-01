# Core: architecture

`mu2e-fastsim-core` (package `mu2efastsim`) holds everything the detector stages share: the physics primitives, the
passive world, table I/O and statistics, the CRY generator backends, the compute-endpoint adapters and the Mu2eBO kit,
and the test tooling of all four repositories. The detector stages import it; they never copy it.

```
                 ┌──────────────── mu2efastsim (this repository) ────────────────┐
 physics         │ eloss (G4 MuBetheBloch, Urban fluctuations, Birks, deltas)     │
                 │ eloss_jax (JAX twin)   materials   transport (MSC, slab loss)  │
 world           │ world (GDML -> convex pieces)   crossings (ray-box, NumPy/JAX) │
 data            │ tables (dumper layout, (run, subrun, event) keys)  match  stats │
 generators      │ cry_native (CRY 1.7 C++ via ctypes)  cry_device (JAX)  cosmic_cry│
 compute         │ endpoints: slurm_ssh, eaf, kit (fastsimkit), parameters, ...    │
 site, tests     │ site (environment-only paths)  inputs (manifest check/fetch)   │
                 │ testing (pytest hooks, site-path checker)  testselect          │
                 └────────────────────────────────────────────────────────────────┘
        ▲                         ▲                          ▲
   crv (stage 1)            tracker (stage 2)          beam (umbrella beam/)
```

## Who owns what

| Area | Modules | Used by |
|---|---|---|
| energy loss and scattering | `eloss`, `eloss_jax`, `materials`, `transport` | CRV and tracker transport, beam (`eloss_jax`) |
| world and crossings | `world`, `worldcensus`, `crossings` | CRV and tracker stage-C transport |
| tables and comparisons | `tables`, `match`, `stats`, `plots` | every fast-vs-stock comparison |
| CRY cosmic generator | `cry_native`, `cry_device`, `cosmic_cry`, `cry_tables`, `cry_validate`, `cry_receipts` | CRV `runchain --gen native|device` (opt-in) |
| public files over HTTP | `cvmfsget` | geometry and conditions on hosts without a CVMFS mount |
| endpoints and the kit | `endpoints/` (`slurm_ssh`, `eaf`, `submission`, `artifacts`, `snapshots`: Mu2e configuration (`mu2e_config`) of the hep-endpoints adapters; `kit`, `parameters`, `geometry`, `chainpoint`) and the job scripts `uva/`, `aurora/`, `iri/`, `eaf/` | Mu2eBO (`fastsimkit`), the throughput campaigns; kit host only (needs hep-endpoints) |
| site configuration | `site`, `inputs` | every command that needs a path, host or account |
| tests | `testing`, `testselect`, `ci/` | the test suites and CI of all four repositories |

Some CRV-specific kit task code still lives in `endpoints/`; it is being moved to the CRV repository.

## Design rules

- NumPy is the reference implementation; JAX twins are opt-in and validated statistically against it (same law,
  not the same random stream). Device precision is float64.
- No site, user or storage path in the code: `mu2efastsim.site` reads the environment and raises `SiteConfigError`
  naming a missing variable; `testing.check_site_paths` keeps the four repositories clean.
- Inputs are Mu2e-internal and never committed; `inputs` verifies a release against a checksum manifest.
- One Mu2eBO point = one job; tasks that read Fermilab-resident inputs in place run only on EAF, other tasks stream
  them into node memory and delete them at exit, and the kit refuses results without the deletion record.
