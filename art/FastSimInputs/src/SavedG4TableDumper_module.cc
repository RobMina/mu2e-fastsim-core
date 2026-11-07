// Export retained G4 products without interpreting trajectories or track endpoints as component exits.
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "TTree.h"
#include "art/Framework/Core/EDAnalyzer.h"
#include "art/Framework/Principal/Event.h"
#include "art/Framework/Services/Registry/ServiceHandle.h"
#include "art_root_io/TFileService.h"
#include "canvas/Utilities/InputTag.h"
#include "cetlib_except/exception.h"
#include "fhiclcpp/types/Atom.h"
#include "fhiclcpp/types/Sequence.h"
#include "Offline/MCDataProducts/inc/CrvStep.hh"
#include "Offline/MCDataProducts/inc/GenParticle.hh"
#include "Offline/MCDataProducts/inc/MCTrajectory.hh"
#include "Offline/MCDataProducts/inc/SimParticle.hh"
#include "Offline/MCDataProducts/inc/StatusG4.hh"
#include "Offline/MCDataProducts/inc/StepPointMC.hh"
#include "Offline/MCDataProducts/inc/SurfaceStep.hh"

namespace mu2e {
namespace {
Long64_t key(unsigned long long value) {
  if (value > static_cast<unsigned long long>(std::numeric_limits<Long64_t>::max()))
    throw cet::exception("SavedG4TableDumper") << "Identity exceeds signed 64-bit schema\n";
  return static_cast<Long64_t>(value);
}
template <class V> void xyz(double (&out)[3], V const& v) {
  out[0] = v.x(); out[1] = v.y(); out[2] = v.z();
}
}

class SavedG4TableDumper : public art::EDAnalyzer {
 public:
  struct Config {
    fhicl::Sequence<art::InputTag> simParticles{fhicl::Name("simParticles"), std::vector<art::InputTag>{}};
    fhicl::Sequence<art::InputTag> genParticles{fhicl::Name("genParticles"), std::vector<art::InputTag>{}};
    fhicl::Sequence<art::InputTag> crvSteps{fhicl::Name("crvSteps"), std::vector<art::InputTag>{}};
    fhicl::Sequence<art::InputTag> stepPoints{fhicl::Name("stepPoints"), std::vector<art::InputTag>{}};
    fhicl::Sequence<art::InputTag> trajectories{fhicl::Name("trajectories"), std::vector<art::InputTag>{}};
    fhicl::Sequence<art::InputTag> surfaces{fhicl::Name("surfaces"), std::vector<art::InputTag>{}};
    fhicl::Sequence<art::InputTag> status{fhicl::Name("status"), std::vector<art::InputTag>{}};
    fhicl::Atom<bool> allowMissing{fhicl::Name("allowMissing"), false};
  };
  using Parameters = art::EDAnalyzer::Table<Config>;
  explicit SavedG4TableDumper(Parameters const& p) : art::EDAnalyzer(p), conf_(p()) {
    for (auto const& tags : {conf_.simParticles(), conf_.genParticles(), conf_.crvSteps(), conf_.stepPoints(),
                             conf_.trajectories(), conf_.surfaces(), conf_.status()}) {
      std::set<std::string> seen;
      for (auto const& tag : tags)
        if (tag.label().empty() || !seen.insert(tag.encode()).second)
          throw cet::exception("SavedG4TableDumper") << "Empty or repeated input tag\n";
    }
  }
  void beginJob() override;
  void analyze(art::Event const&) override;

 private:
  template <class T> art::Handle<T> get(art::Event const&, art::InputTag const&, char const*);
  template <class T> void pointer(art::Ptr<T> const& p, Long64_t& product, Long64_t& index, bool& available) {
    product = p.isNonnull() ? key(p.id().value()) : 0;
    index = p.isNonnull() ? key(p.key()) : -1;
    available = p.isNonnull() && p.isAvailable();
  }
  void particlePointer(art::Ptr<SimParticle> const& p) {
    pointer(p, simProduct_, simKey_, available_);
    if (!available_) throw cet::exception("SavedG4TableDumper") << "Unresolvable response particle pointer\n";
  }
  TTree* table(char const*, char const*, bool source = true);
  void positions(TTree*, bool post, bool momentum);
  void simIdentity(TTree*);
  Config conf_;
  TTree *events_, *collections_, *particles_, *daughters_, *generators_, *crv_, *steps_, *trajs_, *points_, *surfaces_, *status_;
  double endMom_{};
  Long64_t run_{}, subrun_{}, event_{}, product_{}, index_{}, count_{}, schema_{1}, nCollections_{}, nMissing_{};
  std::string tag_, kind_;
  bool present_{}, completeBoundary_{false};
  Long64_t simProduct_{}, simKey_{}, parentProduct_{}, parentKey_{}, genProduct_{}, genKey_{};
  Long64_t daughterProduct_{}, daughterKey_{}, daughterIndex_{};
  bool available_{}, parentAvailable_{}, genAvailable_{}, daughterAvailable_{};
  Long64_t pdg_{}, stage_{}, startVolume_{}, endVolume_{}, startStatus_{}, endStatus_{}, creation_{}, stopping_{}, nSteps_{};
  bool endDefined_{}, truncated_{}, selfParent_{};
  double x0_[3]{}, x1_[3]{}, p0_[3]{}, p1_[3]{}, t0_{}, t1_{}, tau0_{}, tau1_{}, e0_{}, e1_{};
  double length_{}, edep_{}, nonion_{}, visible_{}, endKE_{}, preLastKE_{}, excitation_{};
  Long64_t floatLevel_{}, volume_{}, process_{}, genId_{}, pointIndex_{}, nPoints_{}, surfaceId_{}, surfaceIndex_{};
  double ke_{};
  Long64_t g4Status_{}, nTracks_{}, nKilledSteps_{}, nKilledField_{};
  bool overflow_{};
};

TTree* SavedG4TableDumper::table(char const* name, char const* title, bool source) {
  art::ServiceHandle<art::TFileService> fs;
  auto* t = fs->make<TTree>(name, title);
  t->Branch("run", &run_); t->Branch("subrun", &subrun_); t->Branch("event", &event_);
  if (source) {
    t->Branch("product_id", &product_); t->Branch("tag", &tag_); t->Branch("index", &index_);
  }
  return t;
}
void SavedG4TableDumper::positions(TTree* t, bool post, bool momentum) {
  t->Branch("x0", x0_, "x0[3]/D"); t->Branch("t0", &t0_);
  if (post) t->Branch("x1", x1_, "x1[3]/D");
  if (momentum) t->Branch("p0", p0_, "p0[3]/D");
}
void SavedG4TableDumper::simIdentity(TTree* t) {
  t->Branch("sim_product_id", &simProduct_); t->Branch("sim_key", &simKey_);
}
void SavedG4TableDumper::beginJob() {
  events_ = table("events", "One row per successfully exported input event", false);
  events_->Branch("schema_version", &schema_);
  events_->Branch("n_collections", &nCollections_); events_->Branch("n_missing", &nMissing_);
  events_->Branch("complete_boundary_records", &completeBoundary_);
  collections_ = table("collections", "Configured collection census; missing count is -1", false);
  collections_->Branch("kind", &kind_); collections_->Branch("tag", &tag_);
  collections_->Branch("product_id", &product_); collections_->Branch("present", &present_);
  collections_->Branch("count", &count_);
  particles_ = table("particles", "Stored lifetime states, not matched component boundary states");
  simIdentity(particles_); positions(particles_, true, true);
  particles_->Branch("p1", p1_, "p1[3]/D");
  particles_->Branch("parent_product_id", &parentProduct_); particles_->Branch("parent_key", &parentKey_);
  particles_->Branch("parent_available", &parentAvailable_);
  particles_->Branch("gen_product_id", &genProduct_); particles_->Branch("gen_key", &genKey_);
  particles_->Branch("gen_available", &genAvailable_);
  particles_->Branch("pdg", &pdg_); particles_->Branch("stage", &stage_);
  particles_->Branch("start_volume", &startVolume_); particles_->Branch("end_volume", &endVolume_);
  particles_->Branch("start_status", &startStatus_); particles_->Branch("end_status", &endStatus_);
  particles_->Branch("creation", &creation_); particles_->Branch("stopping", &stopping_);
  particles_->Branch("end_defined", &endDefined_); particles_->Branch("truncated", &truncated_);
  particles_->Branch("self_parent", &selfParent_); particles_->Branch("n_steps", &nSteps_);
  particles_->Branch("length", &length_); particles_->Branch("t1", &t1_);
  particles_->Branch("tau0", &tau0_); particles_->Branch("tau1", &tau1_);
  particles_->Branch("e0", &e0_); particles_->Branch("e1", &e1_);
  particles_->Branch("end_ke", &endKE_); particles_->Branch("pre_last_ke", &preLastKE_);
  particles_->Branch("start_excitation", &excitation_); particles_->Branch("float_level", &floatLevel_);
  daughters_ = table("daughters", "Retained daughter links; not a completeness guarantee");
  simIdentity(daughters_);
  daughters_->Branch("daughter_index", &daughterIndex_);
  daughters_->Branch("daughter_product_id", &daughterProduct_); daughters_->Branch("daughter_key", &daughterKey_);
  daughters_->Branch("daughter_available", &daughterAvailable_);
  generators_ = table("generators", "Configured generator collections without species selection");
  positions(generators_, false, true);
  generators_->Branch("pdg", &pdg_); generators_->Branch("gen_id", &genId_); generators_->Branch("e0", &e0_);
  generators_->Branch("tau0", &tau0_);
  crv_ = table("crv_steps", "Aggregated scintillator response; visible energy is not raw deposition");
  simIdentity(crv_); positions(crv_, true, true);
  crv_->Branch("bar", &volume_); crv_->Branch("t1", &t1_); crv_->Branch("visible", &visible_);
  crv_->Branch("end_momentum_magnitude", &endMom_); crv_->Branch("length", &length_);
  steps_ = table("step_points", "Stored StepPointMC fields; no post time or boundary flag exists");
  simIdentity(steps_); positions(steps_, true, true);
  steps_->Branch("p1", p1_, "p1[3]/D"); steps_->Branch("tau0", &tau0_);
  steps_->Branch("volume", &volume_); steps_->Branch("process", &process_);
  steps_->Branch("length", &length_); steps_->Branch("edep", &edep_);
  steps_->Branch("nonion", &nonion_); steps_->Branch("visible", &visible_);
  trajs_ = table("trajectories", "Retained trajectory headers including empty trajectories");
  simIdentity(trajs_); trajs_->Branch("n_points", &nPoints_);
  points_ = table("trajectory_points", "Stored thinned points; no momentum direction or deposits");
  simIdentity(points_); positions(points_, false, false);
  points_->Branch("point_index", &pointIndex_); points_->Branch("ke", &ke_);
  surfaces_ = table("surface_steps", "Aggregated surfaces in detector coordinates, not Mu2e coordinates");
  simIdentity(surfaces_); positions(surfaces_, true, true);
  surfaces_->Branch("surface_id", &surfaceId_); surfaces_->Branch("surface_index", &surfaceIndex_);
  surfaces_->Branch("edep", &edep_);
  status_ = table("status", "Saved G4 integrity counters");
  status_->Branch("status", &g4Status_); status_->Branch("n_tracks", &nTracks_);
  status_->Branch("overflow", &overflow_); status_->Branch("killed_step_limit", &nKilledSteps_);
  status_->Branch("killed_field", &nKilledField_);
}

template <class T> art::Handle<T> SavedG4TableDumper::get(art::Event const& e, art::InputTag const& tag, char const* kind) {
  art::Handle<T> h;
  e.getByLabel(tag, h);
  tag_ = tag.encode(); kind_ = kind; present_ = h.isValid(); index_ = 0;
  product_ = present_ ? key(h.id().value()) : 0;
  count_ = -1;
  ++nCollections_;
  if (!present_) {
    ++nMissing_;
    if (!conf_.allowMissing())
      throw cet::exception("SavedG4TableDumper") << e.id() << " missing " << kind << " " << tag << '\n';
    collections_->Fill();
  }
  return h;
}

void SavedG4TableDumper::analyze(art::Event const& e) {
  run_ = e.run(); subrun_ = e.subRun(); event_ = e.event(); nCollections_ = nMissing_ = 0;
  for (auto const& tag : conf_.simParticles()) {
    auto h = get<SimParticleCollection>(e, tag, "particles");
    if (!h.isValid()) continue;
    count_ = key(h->size()); collections_->Fill();
    for (auto const& item : *h) {
      auto const& p = item.second;
      simProduct_ = product_; simKey_ = key(item.first.asUint());
      pointer(p.parent(), parentProduct_, parentKey_, parentAvailable_);
      pointer(p.genParticle(), genProduct_, genKey_, genAvailable_);
      pdg_ = p.pdgId(); stage_ = p.simStage();
      startVolume_ = p.startVolumeIndex(); startStatus_ = p.startG4Status();
      creation_ = p.creationCode().id(); stopping_ = p.stoppingCode().id();
      endDefined_ = p.endDefined(); truncated_ = p.isTruncated(); selfParent_ = p.selfParent();
      xyz(x0_, p.startPosition()); xyz(p0_, p.startMomentum().vect());
      t0_ = p.startGlobalTime(); tau0_ = p.startProperTime(); e0_ = p.startMomentum().e();
      excitation_ = p.startExcitationEnergy(); floatLevel_ = p.startFloatLevelBaseIndex();
      const auto nan = std::numeric_limits<double>::quiet_NaN();
      for (unsigned i = 0; i != 3; ++i) x1_[i] = p1_[i] = nan;
      t1_ = tau1_ = e1_ = endKE_ = preLastKE_ = length_ = nan;
      endVolume_ = endStatus_ = nSteps_ = -1;
      if (endDefined_) {
        xyz(x1_, p.endPosition()); xyz(p1_, p.endMomentum().vect());
        t1_ = p.endGlobalTime(); tau1_ = p.endProperTime(); e1_ = p.endMomentum().e();
        endVolume_ = p.endVolumeIndex(); endStatus_ = p.endG4Status();
        endKE_ = p.endKineticEnergy(); preLastKE_ = p.preLastStepKineticEnergy();
        nSteps_ = p.nSteps(); length_ = p.trackLength();
      }
      particles_->Fill(); daughterIndex_ = 0;
      for (auto const& d : p.daughters()) {
        pointer(d, daughterProduct_, daughterKey_, daughterAvailable_);
        daughters_->Fill(); ++daughterIndex_;
      }
      ++index_;
    }
  }
  for (auto const& tag : conf_.genParticles()) {
    auto h = get<GenParticleCollection>(e, tag, "generators");
    if (!h.isValid()) continue;
    count_ = key(h->size()); collections_->Fill();
    for (auto const& g : *h) {
      xyz(x0_, g.position()); xyz(p0_, g.momentum().vect()); t0_ = g.time();
      pdg_ = g.pdgId(); genId_ = g.generatorId().id(); e0_ = g.momentum().e(); tau0_ = g.properTime();
      generators_->Fill(); ++index_;
    }
  }
  for (auto const& tag : conf_.crvSteps()) {
    auto h = get<CrvStepCollection>(e, tag, "crv_steps");
    if (!h.isValid()) continue;
    count_ = key(h->size()); collections_->Fill();
    for (auto const& s : *h) {
      particlePointer(s.simParticle());
      xyz(x0_, s.startPos()); xyz(x1_, s.endPos()); xyz(p0_, s.startMom());
      t0_ = s.startTime(); t1_ = s.endTime(); visible_ = s.visibleEDep();
      volume_ = s.barIndex().asInt(); endMom_ = s.endMom(); length_ = s.pathLength();
      crv_->Fill(); ++index_;
    }
  }
  for (auto const& tag : conf_.stepPoints()) {
    auto h = get<StepPointMCCollection>(e, tag, "step_points");
    if (!h.isValid()) continue;
    count_ = key(h->size()); collections_->Fill();
    for (auto const& s : *h) {
      particlePointer(s.simParticle());
      xyz(x0_, s.position()); xyz(x1_, s.postPosition()); xyz(p0_, s.momentum()); xyz(p1_, s.postMomentum());
      t0_ = s.time(); tau0_ = s.properTime(); volume_ = key(s.volumeId()); process_ = s.endProcessCode().id();
      length_ = s.stepLength(); edep_ = s.totalEDep(); nonion_ = s.nonIonizingEDep(); visible_ = s.visibleEDep();
      steps_->Fill(); ++index_;
    }
  }
  for (auto const& tag : conf_.trajectories()) {
    auto h = get<MCTrajectoryCollection>(e, tag, "trajectories");
    if (!h.isValid()) continue;
    count_ = key(h->size()); collections_->Fill();
    for (auto const& item : *h) {
      auto const& trajectory = item.second;
      particlePointer(trajectory.sim());
      if (item.first != trajectory.sim())
        throw cet::exception("SavedG4TableDumper") << e.id() << " Trajectory map/payload pointer mismatch\n";
      nPoints_ = key(trajectory.size()); trajs_->Fill(); pointIndex_ = 0;
      for (auto const& p : trajectory.points()) {
        xyz(x0_, p.pos()); t0_ = p.t(); ke_ = p.kineticEnergy(); points_->Fill(); ++pointIndex_;
      }
      ++index_;
    }
  }
  for (auto const& tag : conf_.surfaces()) {
    auto h = get<SurfaceStepCollection>(e, tag, "surface_steps");
    if (!h.isValid()) continue;
    count_ = key(h->size()); collections_->Fill();
    for (auto const& s : *h) {
      particlePointer(s.simParticle());
      xyz(x0_, s.startPosition()); xyz(x1_, s.endPosition()); xyz(p0_, s.momentum());
      t0_ = s.time(); edep_ = s.energyDeposit();
      surfaceId_ = s.surfaceId().id().id(); surfaceIndex_ = s.surfaceId().index();
      surfaces_->Fill(); ++index_;
    }
  }
  for (auto const& tag : conf_.status()) {
    auto h = get<StatusG4>(e, tag, "status");
    if (!h.isValid()) continue;
    count_ = 1; collections_->Fill(); g4Status_ = h->status(); nTracks_ = h->nG4Tracks();
    overflow_ = h->overflowSimParticles(); nKilledSteps_ = h->nKilledStepLimit(); nKilledField_ = h->nKilledByFieldPropagator();
    status_->Fill();
  }
  events_->Fill();
}
}

DEFINE_ART_MODULE(mu2e::SavedG4TableDumper)
