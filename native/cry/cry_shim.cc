// SPDX-License-Identifier: Apache-2.0
#include "CRYGenerator.h"
#include "CRYParticle.h"
#include "CRYSetup.h"
#include <cmath>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Particle {
  std::uint64_t shower;
  std::int32_t pdg;
  std::int32_t reserved;
  double ke, x, y, z, u, v, w, t;
};
static_assert(sizeof(Particle) == 80, "CRY particle ABI size");
struct State {
  std::mt19937_64 rng;
  std::unique_ptr<CRYSetup> setup;
  std::unique_ptr<CRYGenerator> generator;
  std::vector<Particle> particles;
  std::vector<double> times;
  explicit State(std::uint64_t seed) : rng(seed) {}
};
std::mutex gate;
thread_local State* active = nullptr;
thread_local std::string error;
struct Active {
  explicit Active(State* state) { active = state; }
  ~Active() { active = nullptr; }
};
double uniform() {
  return (static_cast<double>(active->rng() >> 12) + 0.5) * 0x1p-52;
}
struct Shower {
  std::vector<CRYParticle*> particles;
  ~Shower() { for (auto* p : particles) delete p; }
};
}

extern "C" {
int m2cry_abi_version() noexcept { return 1; }
const char* m2cry_last_error() noexcept { return error.c_str(); }

void* m2cry_create(const char* data, std::uint64_t seed, double subbox,
                  double latitude, int altitude, int month, int day, int year) noexcept {
  std::lock_guard<std::mutex> lock(gate);
  error.clear();
  try {
    if (!data || !*data) throw std::invalid_argument("CRY data directory is required");
    if (!std::isfinite(subbox) || subbox <= 0 || subbox > 300)
      throw std::invalid_argument("CRY subbox must be in (0,300] m");
    if (!std::isfinite(latitude) || latitude < -90 || latitude > 90)
      throw std::invalid_argument("CRY latitude must be in [-90,90] degrees");
    if (altitude != 0 && altitude != 2100 && altitude != 11300)
      throw std::invalid_argument("CRY altitude must be 0,2100,11300 m");
    const int lengths[] = {31,28,31,30,31,30,31,31,30,31,30,31};
    bool leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    if (year < 1900 || year > 9999 || month < 1 || month > 12 || day < 1 ||
        day > lengths[month-1] + (month == 2 && leap))
      throw std::invalid_argument("invalid CRY date");
    for (int h : {0,2100,11300}) {
      std::ifstream stream(std::string(data) + "/cosmics_" + std::to_string(h) + ".data");
      if (!stream.good()) throw std::runtime_error("missing/unreadable CRY altitude table");
    }
    auto state = std::make_unique<State>(seed);
    std::ostringstream config;
    config << std::setprecision(17)
      << "returnMuons 1 returnNeutrons 1 returnProtons 1 returnGammas 1 "
      << "returnElectrons 1 returnPions 1 returnKaons 1 subboxLength " << subbox
      << " latitude " << latitude << " altitude " << altitude
      << " date " << month << "-" << day << "-" << year << " ";
    Active guard(state.get());
    state->setup = std::make_unique<CRYSetup>(config.str(), data);
    state->setup->setRandomFunction(uniform);
    state->generator = std::make_unique<CRYGenerator>(state->setup.get());
    return state.release();
  } catch (const std::exception& e) { error = e.what(); }
    catch (...) { error = "unknown CRY initialisation exception"; }
  return nullptr;
}

int m2cry_generate(void* handle, std::uint64_t count, const Particle** particles,
                   std::uint64_t* size, const double** times, double* before) noexcept {
  std::lock_guard<std::mutex> lock(gate);
  error.clear();
  try {
    if (!handle || !particles || !size || !times || !before)
      throw std::invalid_argument("null CRY argument");
    auto* state = static_cast<State*>(handle);
    state->particles.clear();
    state->times.clear();
    *particles = nullptr;
    *times = nullptr;
    *size = 0;
    *before = state->generator->timeSimulated();
    Active guard(state);
    for (std::uint64_t i = 0; i < count; ++i) {
      Shower shower;
      state->generator->genEvent(&shower.particles);
      for (auto* p : shower.particles)
        state->particles.push_back({i, p->PDGid(), 0, p->ke(), p->x(), p->y(), p->z(),
                                    p->u(), p->v(), p->w(), p->t()});
      state->times.push_back(state->generator->timeSimulated());
    }
    *particles = state->particles.data();
    *size = state->particles.size();
    *times = state->times.data();
    return 0;
  } catch (const std::exception& e) { error = e.what(); }
    catch (...) { error = "unknown CRY generation exception"; }
  return -1;
}

void m2cry_destroy(void* handle) noexcept {
  std::lock_guard<std::mutex> lock(gate);
  delete static_cast<State*>(handle);
}
}
