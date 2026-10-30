"""JAX CRY 1.7 correlated shower sampling and stock Mu2e selection on the device.

Host code parses/uploads pinned PDF tables; it never draws showers for this backend.
Model reference: LLNL UCRL-CODE-227323; notices in docs/third-party/cry.txt.
"""
from functools import partial
from typing import NamedTuple
import time

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pandas as pd

from . import cry_tables
from .cosmic_cry import CosmicSource, MASS

jax.config.update("jax_enable_x64", True)


class DevicePdf(NamedTuple):
    cdf: object
    valid: object
    low: object
    high: object
    mode: object


def _pdf(pdf):
    values = pdf.weights
    total = values.sum(axis=-1, keepdims=True)
    safe = np.where(total > 0, values / np.maximum(total, 1e-300), 1/values.shape[-1])
    cdf = np.cumsum(safe, axis=-1)
    cdf[:, -1] = 1
    lo, hi = np.log10([pdf.low, pdf.high]) if pdf.kind == "log" else (pdf.low, pdf.high)
    return DevicePdf(jnp.asarray(cdf), jnp.asarray(total[:, 0] > 0), jnp.asarray(lo), jnp.asarray(hi),
                     jnp.asarray({"dis": 0, "lin": 1, "log": 2}[pdf.kind]))


def _stack(pdfs):
    fields = list(zip(*[_pdf(pdf) for pdf in pdfs]))
    cdf, valid, lo, hi, mode = (jnp.stack(x) for x in fields)
    return DevicePdf(cdf.reshape(-1, cdf.shape[-1]), valid.reshape(-1), lo, hi, mode)


def inverse(cdf, row, uniform):
    """Per-lane upper-bound search without materialising a PDF row per particle."""
    n = cdf.shape[-1]
    def step(_, state):
        lo, hi = state
        mid = (lo+hi)//2
        less = cdf[row, jnp.minimum(mid, n-1)] <= uniform
        return jnp.where(less, mid+1, lo), jnp.where(less, hi, mid)
    lo = jnp.zeros_like(row)
    hi = jnp.full_like(row, n)
    return lax.fori_loop(0, n.bit_length(), step, (lo, hi))[0]


def draw(pdf, row, u, within, species=None):
    index = inverse(pdf.cdf, row, u)
    lo, hi, mode = (x if species is None else x[species] for x in (pdf.low, pdf.high, pdf.mode))
    n = pdf.cdf.shape[-1]
    linear = lo + (index+within)*(hi-lo)/n
    value = jnp.where(mode == 0, lo + index*(hi-lo)/max(1, n-1),
                      jnp.where(mode == 2, 10.0**linear, linear))
    return value, pdf.valid[row] & (index < n)


def device_tables(path=None):
    prepared = cry_tables.load(path)
    t = prepared["tables"]
    arrays = {"primary": _pdf(prepared["primary_pdf"]), "multiplicity": _pdf(t["nParticles300"]),
              "species": _pdf(t["particleFractions"]), "primary_edges": jnp.asarray(t["primaryBins"]),
              "secondary_edges": jnp.asarray(t["secondaryBins"]),
              "lifetime": jnp.asarray(prepared["lifetime_s"]), "pdg": jnp.asarray(cry_tables.PDG)}
    for key, suffix in (("energy", "KEDist"), ("time", "TimeDist"), ("angle", "CosThetaDist"), ("charge", "ChargeDist")):
        arrays[key] = _stack([t[s+suffix] for s in cry_tables.SPECIES])
    return arrays, {k: prepared[k] for k in ("table_sha256", "date", "latitude", "subbox_m", "lifetime_s")}


def uniform(key, tag, shape):
    return jax.random.uniform(jax.random.fold_in(key, tag), shape, dtype=jnp.float64,
                              minval=2.0**-53, maxval=1.0)


@partial(jax.jit, static_argnames=("showers", "capacity"))
def raw(t, key, before, *, showers, capacity):
    """All raw-shower sampling on device; pool overflow/undefined PDF is returned, never truncated silently."""
    primary, good = draw(t["primary"], jnp.zeros(showers, jnp.int32), uniform(key, 0, (showers,)),
                          uniform(key, 1, (showers,)))
    pbin = jnp.searchsorted(t["primary_edges"], primary, side="right")-1
    mult, ok = draw(t["multiplicity"], pbin, uniform(key, 2, (showers,)), jnp.zeros(showers))
    mult = mult.astype(jnp.int32)
    ends = jnp.cumsum(mult)
    count = ends[-1]
    lane = jnp.arange(capacity)
    shower = jnp.minimum(jnp.searchsorted(ends, lane, side="right"), showers-1)
    mask = lane < count
    pb = pbin[shower]
    species, valid = draw(t["species"], pb, uniform(key, 3, (capacity,)), jnp.zeros(capacity))
    species = species.astype(jnp.int32)-1
    ke, energy_ok = draw(t["energy"], species*(t["primary_edges"].size-1)+pb,
                         uniform(key, 4, (capacity,)), uniform(key, 5, (capacity,)), species)
    sb = jnp.searchsorted(t["secondary_edges"], ke, side="right")-1
    time_row = species*(t["secondary_edges"].size-1)+sb
    delay, time_ok = draw(t["time"], time_row, uniform(key, 6, (capacity,)), uniform(key, 7, (capacity,)), species)
    w, angle_ok = draw(t["angle"], time_row, uniform(key, 8, (capacity,)), uniform(key, 9, (capacity,)), species)
    charge, charge_ok = draw(t["charge"], time_row, uniform(key, 10, (capacity,)), jnp.zeros(capacity), species)
    phi = 2*jnp.pi*uniform(key, 11, (capacity,))
    transverse = jnp.sqrt(jnp.maximum(1-w*w, 0))
    direction = jnp.stack([transverse*jnp.cos(phi), transverse*jnp.sin(phi), w], axis=-1)
    position = jnp.stack([(uniform(key, 12, (capacity,))-.5)*300,
                          (uniform(key, 13, (capacity,))-.5)*300, jnp.zeros(capacity)], axis=-1)
    unsigned = t["pdg"][species]
    negative = jnp.where((unsigned == 11) | (unsigned == 13), charge > 0, charge < 0)
    pdg = jnp.where(negative, -unsigned, unsigned)
    clocks = before + jnp.cumsum(-t["lifetime"]*jnp.log(uniform(key, 14, (showers,))))
    bad = jnp.any(~good | ~ok) | jnp.any(mask & ~(valid & energy_ok & time_ok & angle_ok & charge_ok))
    return {"shower": shower, "mask": mask, "pdg": pdg, "ke": ke, "position": position,
            "direction": direction, "t": clocks[shower]+delay, "clocks": clocks,
            "primary_bin": pbin, "multiplicity": mult, "particles": count,
            "overflow": count > capacity, "invalid_pdf": bad}


def intersection(position, direction, lower, upper):
    """Five full-line faces; fixed axis loop is unrolled during tracing."""
    result = jnp.zeros_like(position)
    highest = jnp.full(position.shape[0], -jnp.inf)
    for axis, plane in ((1, upper[1]), (2, lower[2]), (2, upper[2]), (0, lower[0]), (0, upper[0])):
        scale = (plane-position[:, axis])/jnp.where(direction[:, axis] != 0, direction[:, axis], 1)
        point = (position + scale[:, None]*direction).at[:, axis].set(plane)
        other = np.array([i for i in range(3) if i != axis])
        hit = ((direction[:, axis] != 0) & jnp.all(point[:, other] >= lower[other], axis=1)
               & jnp.all(point[:, other] <= upper[other], axis=1) & (point[:, 1] > highest))
        result = jnp.where(hit[:, None], point, result)
        highest = jnp.where(hit, point[:, 1], highest)
    return result, jnp.isfinite(highest)


@partial(jax.jit, static_argnames=("showers", "selected_capacity"))
def select(r, cfg, key, pending, pending_count, *, showers, selected_capacity):
    """Stock selection/grouping of a device raw batch; exposed separately for deterministic parity tests."""
    mass_index = jnp.searchsorted(cfg["mass_pdg"], jnp.abs(r["pdg"]))
    mass = cfg["mass"][mass_index]
    energy = r["ke"]+mass
    above = r["mask"] & (r["ke"] >= cfg["min_ke"])
    sums = jnp.zeros(showers).at[r["shower"]].add(jnp.where(above, energy, 0))
    position = r["position"][:, jnp.array([1, 2, 0])]*1000+cfg["reference"]
    direction = r["direction"][:, jnp.array([1, 2, 0])]
    _, hit = intersection(position, direction, cfg["target_lo"], cfg["target_hi"])
    position, world_hit = intersection(position, direction, cfg["world_lo"], cfg["world_hi"])
    keep = above & hit
    selected = jnp.sum(keep)
    indices = jnp.nonzero(keep, size=selected_capacity, fill_value=0)[0]
    rows = jnp.concatenate([r["pdg"][:, None], energy[:, None], position,
                            jnp.sqrt(jnp.maximum(energy**2-mass**2, 0))[:, None]*direction,
                            (r["t"]-r["clocks"][r["shower"]])[:, None]], axis=1)[indices]
    low = sums < cfg["max_shower"]
    terminal = lax.associative_scan(jnp.minimum, jnp.where(low, jnp.arange(showers), showers), reverse=True)
    targets = terminal[r["shower"][indices]]
    rows = jnp.concatenate([pending, rows])
    targets = jnp.concatenate([jnp.full(selected_capacity, terminal[0]), targets])
    valid = jnp.concatenate([jnp.arange(selected_capacity) < pending_count,
                             jnp.arange(selected_capacity) < selected])
    complete = valid & (targets < showers)
    counts = jnp.zeros(showers+1, jnp.int32).at[targets].add(complete.astype(jnp.int32))
    event = counts[:-1] > 0
    ranks = jnp.cumsum(event)-1
    slot = ranks[jnp.minimum(targets, showers-1)]
    offsets = (cfg["offset_lo"] + uniform(key, 15, (showers,))*(cfg["offset_hi"]-cfg["offset_lo"])).astype(jnp.float32)
    timed = rows.at[:, -1].add(offsets[jnp.minimum(targets, showers-1)].astype(jnp.float64))
    output_index = jnp.nonzero(complete, size=selected_capacity, fill_value=0)[0]
    pending_mask = valid & (targets == showers)
    pending_index = jnp.nonzero(pending_mask, size=selected_capacity, fill_value=0)[0]
    clock_index = jnp.nonzero(event, size=selected_capacity, fill_value=0)[0]
    output_count, n_pending = jnp.sum(complete), jnp.sum(pending_mask)
    n_events = jnp.sum(event)
    overflow = r["overflow"] | (selected > selected_capacity) | (output_count > selected_capacity) | (n_pending > selected_capacity)
    return {"rows": timed[output_index], "slots": slot[output_index], "row_count": output_count,
            "clocks": r["clocks"][clock_index], "events": n_events, "before_next": r["clocks"][-1],
            "pending": rows[pending_index], "pending_count": n_pending, "particles": r["particles"],
            "overflow": overflow, "invalid_pdf": r["invalid_pdf"],
            "world_miss": jnp.any(keep & ~world_hit), "overenergy": jnp.sum(~low),
            "carried": jnp.sum(keep & (terminal[r["shower"]] != r["shower"]))}


@partial(jax.jit, static_argnames=("showers", "capacity", "selected_capacity"))
def accepted(t, cfg, key, before, pending, pending_count, *, showers, capacity, selected_capacity):
    """Sampling and stock selection execute on the same device; no host shower inputs."""
    r = raw(t, key, before, showers=showers, capacity=capacity)
    return select(r, cfg, key, pending, pending_count, showers=showers, selected_capacity=selected_capacity)


def _config(config):
    pdgs = np.array(sorted(MASS))
    return {"reference": jnp.asarray(config.reference_mm), "target_lo": jnp.asarray(config.target_min_mm),
            "target_hi": jnp.asarray(config.target_max_mm), "world_lo": jnp.asarray(config.world_min_mm),
            "world_hi": jnp.asarray(config.world_max_mm), "min_ke": jnp.asarray(config.min_ke_mev),
            "max_shower": jnp.asarray(config.max_shower_mev), "offset_lo": jnp.asarray(config.offset_ns[0]),
            "offset_hi": jnp.asarray(config.offset_ns[1]), "mass_pdg": jnp.asarray(pdgs),
            "mass": jnp.asarray([MASS[p] for p in pdgs])}


class DeviceCosmicSource(CosmicSource):
    """Device-generated accepted primaries; host buffering is the existing NumPy-chain boundary."""
    def __init__(self, config, seed=1, *, raw_batch=100000, particle_capacity=None, selected_capacity=8192,
                 table=None, require_gpu=True):
        if type(seed) is not int or not 0 <= seed < 2**64 or type(raw_batch) is not int or not 1 <= raw_batch <= 100000:
            raise ValueError("invalid device CRY seed/raw batch")
        if require_gpu and jax.default_backend() != "gpu":
            raise ValueError("on-device CRY requires the GPU backend; CPU is test-only")
        self.capacity = 4*raw_batch if particle_capacity is None else particle_capacity
        if (type(self.capacity) is not int or not 1 <= self.capacity < 2**31
                or type(selected_capacity) is not int or not 1 <= selected_capacity < 2**31):
            raise ValueError("device CRY capacities must be positive integers")
        self.config, self.raw_batch, self.seed = config, raw_batch, seed
        self.selected_capacity = selected_capacity
        start = time.perf_counter()
        self.tables, self.sampling = device_tables(table)
        self.device_config = _config(config)
        self.key = jax.random.fold_in(jax.random.key(seed & 0xffffffff, impl="threefry2x32"), seed >> 32)
        self.pending_device = jnp.zeros((selected_capacity, 9))
        self.pending_count = jnp.asarray(0)
        self.before = jnp.asarray(0.0)
        jax.block_until_ready(self.tables)
        self.setup_s = time.perf_counter()-start
        self.raw_offset = self.overenergy_showers = self.accepted_from_overenergy = self.batches = 0
        self.last_time = self.kernel_wall_s = self.copy_wall_s = 0.0
        self.buffer, self.clocks = pd.DataFrame(), np.empty(0)
        self.closed = False
        self.failed = False
        self.device = next(iter(self.pending_device.devices()))
        if require_gpu and self.device.platform != "gpu":
            raise ValueError("device CRY arrays were placed on a non-GPU default device")

    def _fill(self):
        if self.closed or self.failed:
            raise RuntimeError("device CRY source is closed or failed")
        started = time.perf_counter()
        key = jax.random.fold_in(self.key, self.batches)
        out = accepted(self.tables, self.device_config, key, self.before, self.pending_device, self.pending_count,
                       showers=self.raw_batch, capacity=self.capacity, selected_capacity=self.selected_capacity)
        jax.block_until_ready(out)
        self.kernel_wall_s += time.perf_counter()-started
        started = time.perf_counter()
        host = jax.device_get({k: v for k, v in out.items() if k not in ("pending", "pending_count", "before_next")})
        self.copy_wall_s += time.perf_counter()-started
        if host["overflow"]:
            self.failed = True
            raise RuntimeError("device CRY pool overflow; entire batch refused, enlarge capacity without resampling")
        if host["invalid_pdf"] or host["world_miss"]:
            self.failed = True
            raise RuntimeError("device CRY reached undefined PDF or target particle missed world")
        self.pending_device, self.pending_count, self.before = out["pending"], out["pending_count"], out["before_next"]
        n, events = int(host["row_count"]), int(host["events"])
        frame = pd.DataFrame(host["rows"][:n], columns=["pdg", "e", "x", "y", "z", "px", "py", "pz", "t"])
        frame["slot"] = host["slots"][:n]+len(self.clocks)
        self.buffer = pd.concat([self.buffer, frame], ignore_index=True) if len(self.buffer) else frame
        self.clocks = np.concatenate([self.clocks, host["clocks"][:events]])
        self.raw_offset += self.raw_batch
        self.overenergy_showers += int(host["overenergy"])
        self.accepted_from_overenergy += int(host["carried"])
        self.batches += 1

    def generate(self, count, **kw):
        if self.closed or self.failed:
            raise RuntimeError("device CRY source is closed or failed")
        gen, subruns, receipt = super().generate(count, **kw)
        receipt.update(generator="jax-cry-1.7", sampling=self.sampling, device=str(self.device),
                       particle_capacity=self.capacity, selected_capacity=self.selected_capacity,
                       device_batches=self.batches, device_setup_s=self.setup_s,
                       device_kernel_wall_s=self.kernel_wall_s, device_copy_wall_s=self.copy_wall_s,
                       rng="jax threefry2x32; independent streams by batch/quantity; not art/CLHEP/mt19937")
        return gen, subruns, receipt

    def close(self):
        self.closed = True
        self.tables = self.pending_device = self.device_config = None
