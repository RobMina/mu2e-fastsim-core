"""mu2efastsim -- detector-independent core of the Mu2e Geant4-free fast simulation (extracted / KPP configuration).

Shared by the CRV (mu2e-crv-fastsim) and tracker (mu2e-tracker-fastsim) stages, which sit beside it as submodules of
the mu2e-fastsim umbrella repository. Definitions live once in these modules, documented in ``README.md``:

    eloss      muon energy loss in thin layers matched to Geant4 11.3.2 (restricted Bethe, Urban fluctuations, Birks,
               delta rays, radiation length, Highland angle)
    materials  material constants matched to what Geant4 uses (GDML composition, density-effect, fluctuation model)
    world      passive world material from a Geant4 GDML export, ray-traced as convex pieces
    crossings  vectorised ray-box crossings of straight tracks (JAX on CPU/GPU, else numpy)
    transport  detector-independent transport pieces: correlated multiple scattering to a plane, sliced slab loss
    tables     flat-table I/O in the art dumper layout (TFileService directory dump/), keyed by (run, subrun, event)
    stats      robust statistics shared by every comparison (median, core sigma, MPV, binned ratio band)
    plots      shared plotting helpers + write_json
"""
__version__ = "0.1.0"
