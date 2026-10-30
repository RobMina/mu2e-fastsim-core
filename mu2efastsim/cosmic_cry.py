"""Stock-style extracted CosmicCRY selection/projection and CosmicTimeOffset, opt-in on the host."""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .cry_native import CrySource
from .cry_receipts import EXTRACTED_V04_GDML_SHA256

MASS = {11: 0.510998910, 13: 105.658367, 22: 0.0, 211: 139.57018,
        321: 493.677, 2112: 939.56536, 2212: 938.27203}
COLUMNS = ["run", "subrun", "event", "idx", "pdg", "genid", "x", "y", "z", "px", "py", "pz", "e", "t"]


@dataclass(frozen=True)
class CosmicConfig:
    reference_mm: tuple
    target_min_mm: tuple
    target_max_mm: tuple
    world_min_mm: tuple
    world_max_mm: tuple
    geometry_sha256: str
    min_ke_mev: float = 50.0
    max_shower_mev: float = 200000.0
    offset_ns: tuple = (0.0, 100005.0)

    def __post_init__(self):
        for lo, hi in ((self.target_min_mm, self.target_max_mm), (self.world_min_mm, self.world_max_mm)):
            if np.shape(lo) != (3,) or np.shape(hi) != (3,) or not np.all(np.isfinite([lo, hi])) or np.any(np.asarray(lo) >= hi):
                raise ValueError("invalid CosmicCRY box bounds")
        if np.shape(self.reference_mm) != (3,) or not np.all(np.isfinite(self.reference_mm)):
            raise ValueError("invalid CosmicCRY reference point")
        if not 0 <= self.min_ke_mev < self.max_shower_mev or not np.isfinite(self.max_shower_mev):
            raise ValueError("invalid CosmicCRY energy cuts")
        if len(self.offset_ns) != 2 or not np.all(np.isfinite(self.offset_ns)) or self.offset_ns[0] >= self.offset_ns[1]:
            raise ValueError("invalid CosmicTimeOffset window")

    @classmethod
    def extracted(cls, gdml):
        """Use the extracted-v04 reference and read its centered world/HallAir boxes from GDML."""
        digest = hashlib.sha256(Path(gdml).read_bytes()).hexdigest()
        if digest != EXTRACTED_V04_GDML_SHA256:
            raise ValueError("unverified extracted-v04 GDML signature; supply an explicit CosmicConfig for other geometry")
        root = ET.parse(gdml).getroot()
        vols = {v.get("name"): v for v in root.find("structure").findall("volume")}
        solids = {s.get("name"): s for s in root.find("solids")}
        world = vols[root.find("setup/world").get("ref")]
        box = solids[world.find("solidref").get("ref")]
        if box.tag != "box" or box.get("lunit", "mm") != "mm":
            raise ValueError("CosmicCRY expects an mm centered world box")
        half = np.array([float(box.get(a)) / 2 for a in "xyz"])
        halls = [p for p in world.findall("physvol") if p.get("name", "").startswith("HallAir")]
        if len(halls) != 1:
            raise ValueError("expected one HallAir placement")
        hall = halls[0]
        if hall.find("rotation") is not None or hall.find("rotationref") is not None:
            raise ValueError("rotated HallAir is not supported")
        hb = solids[vols[hall.find("volumeref").get("ref")].find("solidref").get("ref")]
        pos = hall.find("position")
        if hb.tag != "box" or hb.get("lunit", "mm") != "mm" or pos.get("unit", "mm") != "mm":
            raise ValueError("expected mm HallAir box")
        ymin = float(pos.get("y", "0")) - float(hb.get("y")) / 2
        ymax = float(pos.get("y", "0")) + float(hb.get("y")) / 2
        lower, upper = -half + 1, half - 1
        lower[1], upper[1] = ymin + 1, ymax - 1
        return cls((-3904.0, 20000.0, 24171.0), (-10000.0, -5000.0, 17105.0),
                   (3000.0, 9880.0, 32220.0), tuple(lower), tuple(upper),
                   digest)


def highest_intersection(position, direction, lower, upper):
    """Five-face full-line intersections, inclusive edges, highest y; no floor or forward-ray cut."""
    position, direction = np.asarray(position, float), np.asarray(direction, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    result = np.full_like(position, np.nan)
    highest = np.full(len(position), -np.inf)
    for axis, plane in ((1, upper[1]), (2, lower[2]), (2, upper[2]), (0, lower[0]), (0, upper[0])):
        scale = np.divide(plane - position[:, axis], direction[:, axis],
                          out=np.zeros(len(position)), where=direction[:, axis] != 0)
        point = position + scale[:, None] * direction
        point[:, axis] = plane
        other = [a for a in range(3) if a != axis]
        hit = ((direction[:, axis] != 0) & np.all(point[:, other] >= lower[other], axis=1)
               & np.all(point[:, other] <= upper[other], axis=1))
        use = hit & (point[:, 1] > highest)
        result[use], highest[use] = point[use], point[use, 1]
    return result, np.isfinite(highest)


def project(batch, config):
    """Vectorised stock per-particle cut/map and per-raw-shower energy, before shower acceptance."""
    rows = batch.particles
    rows = rows[rows["ke"] >= config.min_ke_mev]
    pdg = rows["pdg"]
    mass = np.zeros(len(rows))
    known = np.zeros(len(rows), bool)
    for p, m in MASS.items():
        keep = np.abs(pdg) == p
        mass[keep], known[keep] = m, True
    if not np.all(known):
        raise ValueError("unknown CRY particle mass")
    energy = rows["ke"] + mass
    shower = rows["shower"].astype(np.int64)
    sums = np.bincount(shower, weights=energy, minlength=len(batch.cumulative_seconds))
    position = np.column_stack([rows["y"], rows["z"], rows["x"]]) * 1000 + config.reference_mm
    direction = np.column_stack([rows["v"], rows["w"], rows["u"]])
    _, hit = highest_intersection(position, direction, config.target_min_mm, config.target_max_mm)
    position, world_hit = highest_intersection(position[hit], direction[hit], config.world_min_mm, config.world_max_mm)
    if not np.all(world_hit):
        raise RuntimeError("target-intersecting particle misses stock world projection")
    momentum = np.sqrt(energy[hit] * energy[hit] - mass[hit] * mass[hit])[:, None] * direction[hit]
    selected = pd.DataFrame({"shower": shower[hit], "pdg": pdg[hit], "e": energy[hit],
                             "t": rows["t"][hit] - batch.cumulative_seconds[shower[hit]]})
    for j, a in enumerate("xyz"):
        selected[a] = position[:, j]
        selected["p" + a] = momentum[:, j]
    return selected, sums


class CosmicSource:
    """Stock-compatible accepted events, including the over-energy-loop accumulation, generated in batches."""

    def __init__(self, config, seed=1, *, raw_batch=100000, source=None):
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**64:
            raise ValueError("seed must be uint64")
        if not isinstance(raw_batch, int) or isinstance(raw_batch, bool) or not 1 <= raw_batch <= 100000:
            raise ValueError("raw_batch must be in [1,100000]")
        self.config, self.raw_batch, self.seed = config, raw_batch, seed
        self.source = source if source is not None else CrySource(seed=seed)
        self.offset_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x435259]))
        self.raw_offset = 0
        self.last_time = 0.0
        self.pending = pd.DataFrame()
        self.buffer = pd.DataFrame()
        self.clocks = np.empty(0)
        self.overenergy_showers = 0
        self.accepted_from_overenergy = 0

    def _fill(self):
        batch = self.source.next_showers(self.raw_batch)
        selected, sums = project(batch, self.config)
        low = np.flatnonzero(sums < self.config.max_shower_mev)
        self.overenergy_showers += int(np.sum(sums >= self.config.max_shower_mev))
        if len(selected):
            selected["shower"] += self.raw_offset
        if len(self.pending):
            selected = pd.concat([self.pending, selected], ignore_index=True)
        self.pending = pd.DataFrame()
        terminal = low + self.raw_offset
        target = np.searchsorted(terminal, selected.shower.to_numpy()) if len(selected) else np.empty(0, int)
        keep = target < len(terminal)
        self.pending = selected.loc[~keep].copy()
        selected = selected.loc[keep].copy()
        if len(selected):
            selected["terminal"] = terminal[target[keep]]
            accepted = np.unique(selected.terminal)
            self.accepted_from_overenergy += int(np.sum(selected.shower != selected.terminal))
            selected["slot"] = np.searchsorted(accepted, selected.terminal)
            del selected["terminal"]
            offsets = self.offset_rng.uniform(*self.config.offset_ns, size=len(accepted)).astype(np.float32)
            selected["t"] += offsets[selected.slot.to_numpy()].astype(float)
            clocks = batch.cumulative_seconds[accepted - self.raw_offset]
            if len(self.buffer):
                selected["slot"] += len(self.clocks)
                self.buffer = pd.concat([self.buffer, selected], ignore_index=True)
                self.clocks = np.concatenate([self.clocks, clocks])
            else:
                self.buffer, self.clocks = selected, clocks
        self.raw_offset += len(batch.cumulative_seconds)

    def generate(self, count, *, run=1400, subrun=1, first_event=1):
        """Return (gen DataFrame, subruns DataFrame, receipt); no file dependency or per-event Python loop."""
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("accepted-event count must be a positive integer")
        ids = (run, subrun, first_event)
        if any(not isinstance(x, int) or isinstance(x, bool) or not 0 <= x < 2**31 for x in ids):
            raise ValueError("event identifiers must be nonnegative int32")
        if first_event < 1 or first_event + count > 2**31:
            raise ValueError("event range overflows int32")
        while len(self.clocks) < count:
            self._fill()
        use = self.buffer.slot.to_numpy() < count
        gen = self.buffer.loc[use].copy()
        self.buffer = self.buffer.loc[~use].copy()
        self.buffer["slot"] -= count
        final_time = float(self.clocks[count-1])
        live = final_time - self.last_time
        self.last_time = final_time
        self.clocks = self.clocks[count:]
        gen["event"] = gen.slot.to_numpy() + first_event
        gen["idx"] = gen.groupby("slot", sort=False).cumcount()
        gen["run"], gen["subrun"], gen["genid"] = run, subrun, 38
        result = gen[COLUMNS].reset_index(drop=True)
        for c in ("run", "subrun", "event", "idx", "pdg", "genid"):
            result[c] = result[c].astype(np.int32)
        subruns = pd.DataFrame({"run": [run], "subrun": [subrun], "livetime": [live], "primaries": [1]})
        receipt = {"generator": "native-cry-1.7", "seed": self.seed, "raw_batch_size": self.raw_batch,
                   "accepted_events": count, "live_seconds": live,
                   "event_range": {"run": run, "subrun": subrun, "first_event": first_event,
                                   "last_event": first_event + count - 1},
                   "raw_showers_drawn": self.raw_offset, "buffered_accepted_events": len(self.clocks),
                   "overenergy_raw_showers_drawn": self.overenergy_showers,
                   "accumulated_particle_rows": self.accepted_from_overenergy,
                   "stock_overenergy_accumulation": True, "cosmic_config": asdict(self.config),
                   "cosmic_time_offset": "one uniform float32 per accepted event; raw seconds retained numerically"}
        return result, subruns, receipt

    def close(self):
        self.source.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
