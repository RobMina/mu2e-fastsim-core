"""Opt-in local interaction replay and vacuum boundary handoff; docs/component-response.md."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

C_MM_NS = 299.792458
POSITION_ATOL, DIRECTION_ATOL = 1e-7, 1e-12
STATE_RTOL, STATE_ATOL = 1e-12, 1e-9
ENTRANCE = dict(pdg=(), position=(3,), momentum=(3,), energy=())
INCOMING = dict(event=(3,), track=(), parent=(), **ENTRANCE, time=())
PARTICLES = dict(pdg=(), parent=(), exited=())
STEPS = dict(particle=(), volume=(), x0=(3,), x1=(3,), p0=(3,), p1=(3,), t0=(), t1=(), length=(), edep=())
EXITS = dict(particle=(), position=(3,), momentum=(3,), energy=(), time=())
INTEGER = {"pdg", "event", "track", "parent", "particle", "volume"}


def _integer(value, name, shape=None):
    a = np.asarray(value)
    if a.dtype.kind not in "iu" or (a.dtype.kind == "u" and np.any(a > np.iinfo(np.int64).max)):
        raise ValueError(f"{name} requires signed-int64-compatible integers")
    a = np.array(a, dtype=np.int64, copy=True)
    if shape is not None and a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return a


def _real(value, name, shape=None):
    a = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(a).all() or (shape is not None and a.shape != shape):
        raise ValueError(f"invalid finite values or shape for {name}")
    return a


def _record(table, schema):
    if set(table) != set(schema):
        raise ValueError(f"expected columns {tuple(schema)}")
    n = len(np.asarray(table[next(iter(schema))]))
    out = {}
    for key, tail in schema.items():
        if key == "exited":
            a = np.asarray(table[key])
            if a.dtype.kind != "b" or a.shape != (n,):
                raise ValueError("exited must be a boolean vector")
            a = a.copy()
        else:
            a = (_integer if key in INTEGER else _real)(table[key], key, (n, *tail))
        a.flags.writeable = False
        out[key] = a
    return MappingProxyType(out)


def _kinematics(table):
    p = np.linalg.norm(table["momentum"], axis=1)
    if np.any(p <= 0) or np.any(table["energy"] < p - STATE_ATOL - STATE_RTOL * p):
        raise ValueError("propagating particles require nonzero momentum and energy >= |p|")
    return p


@dataclass(frozen=True)
class ComponentType:
    """Identical local geometry/material and physics/cuts; field-free box envelope."""
    geometry: str
    physics: str
    half_size: tuple[float, float, float]
    volumes: int

    def __post_init__(self):
        h = _real(self.half_size, "half_size", (3,))
        v = _integer(self.volumes, "volumes", ())
        if not self.geometry or not self.physics or np.any(h <= POSITION_ATOL) or v <= 0:
            raise ValueError("component requires construction/physics labels, positive sizes and volumes")
        object.__setattr__(self, "half_size", tuple(h))
        object.__setattr__(self, "volumes", int(v))

    def inside(self, points):
        return np.all(np.abs(points) <= np.asarray(self.half_size) + POSITION_ATOL, axis=-1)

    def on_boundary(self, points, momentum, *, entering):
        h = np.asarray(self.half_size)
        on = np.abs(np.abs(points) - h) <= POSITION_ATOL
        direction = momentum / np.linalg.norm(momentum, axis=1)[:, None]
        normal_dot = np.sign(points) * direction
        inward = np.all(~on | (normal_dot < -DIRECTION_ATOL), axis=1) & np.any(on, axis=1)
        outward = np.any(on & (normal_dot > DIRECTION_ATOL), axis=1)
        return self.inside(points) & (inward if entering else outward)


@dataclass(frozen=True)
class Placement:
    """A rigidly placed component with receiving global sensitive/material volume IDs."""
    component: ComponentType
    instance: int
    rotation: np.ndarray
    translation: np.ndarray
    volume_ids: np.ndarray

    def __post_init__(self):
        r = _real(self.rotation, "rotation", (3, 3))
        a = _real(self.translation, "translation", (3,))
        ids = _integer(self.volume_ids, "volume_ids", (self.component.volumes,))
        instance = _integer(self.instance, "instance", ())
        if not np.allclose(r.T @ r, np.eye(3), atol=1e-12, rtol=0) or not np.isclose(np.linalg.det(r), 1., atol=1e-12, rtol=0):
            raise ValueError("rotation must be proper orthonormal, not a reflection or scale")
        if instance < 0 or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
            raise ValueError("instance/volume IDs must be nonnegative and volume IDs unique")
        for key, value in (("rotation", r), ("translation", a), ("volume_ids", ids)):
            value.flags.writeable = False
            object.__setattr__(self, key, value)
        object.__setattr__(self, "instance", int(instance))

    def local_points(self, value):
        return (np.asarray(value) - self.translation) @ self.rotation

    def global_points(self, value):
        return np.asarray(value) @ self.rotation.T + self.translation

    def local_vectors(self, value):
        return np.asarray(value) @ self.rotation

    def global_vectors(self, value):
        return np.asarray(value) @ self.rotation.T


def _offsets(offsets, trials, rows, name):
    a = _integer(offsets, name, (trials + 1,))
    if a[0] != 0 or a[-1] != rows or np.any(np.diff(a) < 0):
        raise ValueError(f"invalid offsets for {name}")
    a.flags.writeable = False
    return a


def _trial_rows(offsets):
    return np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))


def _gather(offsets, trials):
    counts = np.diff(offsets)[trials]
    owner = np.repeat(np.arange(len(trials)), counts)
    begin = np.repeat(offsets[trials] - np.r_[0, np.cumsum(counts)[:-1]], counts)
    return owner, begin + np.arange(counts.sum()), counts


class Library:
    """Complete equal-weight trials at recorded local entrance states, including null outcomes."""

    def __init__(self, component, entrances, particles, particle_offsets, steps, step_offsets, exits, exit_offsets):
        self.component = component
        self.entrances = _record(entrances, ENTRANCE)
        self.particles, self.steps, self.exits = (_record(t, s) for t, s in
                                                ((particles, PARTICLES), (steps, STEPS), (exits, EXITS)))
        n = len(self.entrances["pdg"])
        if n == 0:
            raise ValueError("library needs at least one incident trial")
        for name, table, offsets in (("particle", self.particles, particle_offsets),
                                     ("step", self.steps, step_offsets), ("exit", self.exits, exit_offsets)):
            setattr(self, name + "_offsets", _offsets(offsets, n, len(table[next(iter(table))]), name))
        _kinematics(self.entrances)
        _kinematics(self.exits)
        if not component.on_boundary(self.entrances["position"], self.entrances["momentum"], entering=True).all():
            raise ValueError("entrances must be inward on the component boundary")
        if not component.on_boundary(self.exits["position"], self.exits["momentum"], entering=False).all():
            raise ValueError("exits must be outward on the component boundary")
        npart = np.diff(self.particle_offsets)
        if np.any(npart == 0):
            raise ValueError("every trial needs its incident root particle")
        pt = _trial_rows(self.particle_offsets)
        local = np.arange(len(pt)) - self.particle_offsets[pt]
        parents = self.particles["parent"]
        if np.any(np.where(local == 0, parents != -1, (parents < 0) | (parents >= local))):
            raise ValueError("local ancestry must have root0 and parents preceding children")
        if np.any(self.particles["pdg"][self.particle_offsets[:-1]] != self.entrances["pdg"]):
            raise ValueError("root species differs from incident species")
        for table, offsets in ((self.steps, self.step_offsets), (self.exits, self.exit_offsets)):
            owner = _trial_rows(offsets)
            if np.any(table["particle"] < 0) or np.any(table["particle"] >= npart[owner]):
                raise ValueError("particle index outside trial")
        sx = self.steps
        if np.any(sx["volume"] < 0) or np.any(sx["volume"] >= component.volumes):
            raise ValueError("local step volume outside component")
        if not component.inside(sx["x0"]).all() or not component.inside(sx["x1"]).all():
            raise ValueError("step endpoint outside component; no clamping is allowed")
        if np.any(sx["edep"] < 0) or np.any(sx["length"] < 0) or np.any(sx["t0"] < 0) or np.any(sx["t1"] < sx["t0"]) or np.any(self.exits["time"] < 0):
            raise ValueError("negative deposits/lengths or invalid relative times")
        if np.any(sx["length"] + POSITION_ATOL < np.linalg.norm(sx["x1"] - sx["x0"], axis=1)):
            raise ValueError("step length is shorter than its endpoint separation")
        ep = self.particle_offsets[_trial_rows(self.exit_offsets)] + self.exits["particle"]
        exited = np.bincount(ep, minlength=len(pt))
        if np.any(exited != self.particles["exited"].astype(int)):
            raise ValueError("exited particles need exactly one first-exit record")
        exit_time = np.full(len(pt), np.inf)
        exit_time[ep] = self.exits["time"]
        sp = self.particle_offsets[_trial_rows(self.step_offsets)] + sx["particle"]
        if np.any(sx["t1"] > exit_time[sp] + STATE_ATOL):
            raise ValueError("step after first exit is not component-local")


class UnsupportedEntrance(ValueError):
    """Entrance rows outside the bank's explicit state support, never zero-response draws."""

    def __init__(self, rows):
        self.rows = np.asarray(rows, dtype=np.int64)
        super().__init__(f"unsupported local entrance rows: {self.rows.tolist()}")


def _choose(library, local, rng):
    n, donors = len(local["pdg"]), library.entrances
    left, right = [], []
    chunk = max(1, 2_000_000 // len(donors["pdg"]))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        good = local["pdg"][start:stop, None] == donors["pdg"][None, :]
        for key in ("position", "momentum", "energy"):
            a, b = local[key][start:stop], donors[key]
            atol, rtol = (POSITION_ATOL, 0.) if key == "position" else (STATE_ATOL, STATE_RTOL)
            close = np.isclose(a[:, None], b[None, :], atol=atol, rtol=rtol)
            good &= np.all(close, axis=-1) if a.ndim == 2 else close
        ii, jj = np.nonzero(good)
        left.append(ii + start)
        right.append(jj)
    left = np.concatenate(left) if left else np.empty(0, dtype=np.int64)
    right = np.concatenate(right) if right else np.empty(0, dtype=np.int64)
    counts = np.bincount(left, minlength=n)
    if np.any(counts == 0):
        raise UnsupportedEntrance(np.flatnonzero(counts == 0))
    pick = np.cumsum(counts) - counts + (rng.random(n) * counts).astype(np.int64)
    return right[pick]


def _incoming(table):
    incoming = _record(table, INCOMING)
    _kinematics(incoming)
    if np.any(incoming["track"] <= 0) or np.any((incoming["parent"] != -1) & (incoming["parent"] <= 0)):
        raise ValueError("track/parent IDs must be positive or parent=-1")
    if np.any(incoming["parent"] == incoming["track"]):
        raise ValueError("particle cannot parent itself")
    return incoming


@dataclass
class Replay:
    """Correlated transformed particle tree, deposit rows, outgoing states and allocator cursor."""
    particles: dict
    steps: dict
    exits: dict
    trials: np.ndarray
    next_track: int


def replay(library, placement, incoming, interaction, rng, *, next_track):
    """Replay exact supported local entrance states in a rigid placement; no energy/shape interpolation."""
    incoming = _incoming(incoming)
    n = len(incoming["track"])
    interaction = _integer(interaction, "interaction", (n,))
    cursor = int(_integer(next_track, "next_track", ()))
    if placement.component != library.component:
        raise ValueError("incompatible component geometry/material/physics")
    if cursor <= max(incoming["track"].max(initial=0), incoming["parent"].max(initial=0)):
        raise ValueError("next_track must exceed input track/parent IDs and reserve unused IDs")
    key = np.column_stack((incoming["event"], interaction))
    if np.any(interaction < 0) or len(np.unique(key, axis=0)) != n:
        raise ValueError("interaction IDs must be nonnegative and unique within each event")
    local = {**incoming, "position": placement.local_points(incoming["position"]),
             "momentum": placement.local_vectors(incoming["momentum"])}
    valid = library.component.on_boundary(local["position"], local["momentum"], entering=True)
    if not valid.all():
        raise UnsupportedEntrance(np.flatnonzero(~valid))
    trial = _choose(library, local, rng)
    owner, pidx, counts = _gather(library.particle_offsets, trial)
    starts = np.r_[0, np.cumsum(counts)[:-1]] if n else np.empty(0, dtype=np.int64)
    local_id = np.arange(len(pidx)) - starts[owner]
    isroot = local_id == 0
    children = int((~isroot).sum())
    if cursor + children > np.iinfo(np.int64).max:
        raise ValueError("track ID allocator overflow")
    track = np.empty(len(pidx), dtype=np.int64)
    track[isroot] = incoming["track"][owner[isroot]]
    track[~isroot] = cursor + np.arange(children)
    lp = library.particles["parent"][pidx]
    parent = track[starts[owner] + np.maximum(lp, 0)]
    parent[isroot] = incoming["parent"][owner[isroot]]

    def common(ii):
        return dict(event=incoming["event"][ii].copy(), interaction=interaction[ii].copy(),
                    instance=np.full(len(ii), placement.instance, dtype=np.int64))

    particles = dict(**common(owner), track=track, parent=parent,
                     pdg=library.particles["pdg"][pidx].copy(), exited=library.particles["exited"][pidx].copy())
    outputs = []
    for table, offsets, isstep in ((library.steps, library.step_offsets, True), (library.exits, library.exit_offsets, False)):
        ii, idx, _ = _gather(offsets, trial)
        pi = starts[ii] + table["particle"][idx]
        out = dict(**common(ii), track=track[pi], parent=parent[pi], pdg=particles["pdg"][pi])
        if isstep:
            out["volume"] = placement.volume_ids[table["volume"][idx]]
            for name in ("x0", "x1"):
                out[name] = placement.global_points(table[name][idx])
            for name in ("p0", "p1"):
                out[name] = placement.global_vectors(table[name][idx])
            for name in ("t0", "t1"):
                out[name] = table[name][idx] + incoming["time"][ii]
            for name in ("length", "edep"):
                out[name] = table[name][idx].copy()
        else:
            out["position"] = placement.global_points(table["position"][idx])
            out["momentum"] = placement.global_vectors(table["momentum"][idx])
            out["time"] = table["time"][idx] + incoming["time"][ii]
            out["energy"] = table["energy"][idx].copy()
        outputs.append(out)
    return Replay(particles, *outputs, trial, cursor + children)


def next_in_vacuum(particles, placements):
    """Nearest outside-to-box entry in field-free vacuum; return source rows, placement indices and states."""
    particles = _incoming(particles)
    n = len(particles["track"])
    instances = [p.instance for p in placements]
    if len(set(instances)) != len(instances):
        raise ValueError("placement instance IDs must be unique")
    best, which = np.full(n, np.inf), np.full(n, -1, dtype=np.int64)
    ambiguous = np.zeros(n, dtype=bool)
    pmag = np.linalg.norm(particles["momentum"], axis=1)
    for index, placement in enumerate(placements):
        pos = placement.local_points(particles["position"])
        direction = placement.local_vectors(particles["momentum"]) / pmag[:, None]
        h = np.asarray(placement.component.half_size)
        if np.any(np.all(np.abs(pos) < h - POSITION_ATOL, axis=1)):
            raise ValueError("vacuum handoff starts inside a component")
        on = np.abs(np.abs(pos) - h) <= POSITION_ATOL
        pos = np.where(on, np.sign(pos) * h, pos)
        parallel = np.abs(direction) <= DIRECTION_ATOL
        inv = np.divide(1., direction, out=np.zeros_like(direction), where=~parallel)
        a, b = (-h - pos) * inv, (h - pos) * inv
        lo = np.where(parallel, -np.inf, np.minimum(a, b)).max(axis=1)
        hi = np.where(parallel, np.inf, np.maximum(a, b)).min(axis=1)
        miss = np.any(parallel & (np.abs(pos) >= h), axis=1)
        distance = np.maximum(lo, 0.)
        valid = ~miss & (hi > distance + POSITION_ATOL) & (lo >= -POSITION_ATOL)
        distance = np.where(valid, distance, np.inf)
        tie = valid & np.isfinite(best) & (np.abs(distance - np.where(np.isfinite(best), best, 0.)) <= POSITION_ATOL)
        nearer = valid & (distance < best - POSITION_ATOL)
        ambiguous = np.where(nearer, False, ambiguous | tie)
        best, which = np.where(nearer, distance, best), np.where(nearer, index, which)
    if ambiguous.any():
        raise ValueError("ambiguous nearest component entry")
    rows = np.flatnonzero(which >= 0)
    out = {key: a[rows].copy() for key, a in particles.items()}
    out["position"] += particles["momentum"][rows] / pmag[rows, None] * best[rows, None]
    out["time"] += best[rows] * particles["energy"][rows] / (pmag[rows] * C_MM_NS)
    return rows, which[rows], out
