"""EXPERIMENTAL segment-history transport policy; not a production transport replacement.

Uses the existing restricted mean and Urban sampler, without new loss/material physics.
There is no time transport, geometry, scattering, or claim of Geant4 stopping-range agreement.
"""
import numpy as np

from . import eloss, materials


def _positive_scalar(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 0 or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(value)


def _validate_material(mat):
    if not isinstance(mat, materials.Material):
        raise ValueError("mat must be a materials.Material with valid loss constants")
    _positive_scalar(mat.rho, "mat.rho")
    _positive_scalar(mat.I_eV, "mat.I_eV")
    if not mat.fractions or any(e not in materials.ELEMENTS for e in mat.fractions):
        raise ValueError("mat must have a known, nonempty elemental composition")
    weights = np.asarray(list(mat.fractions.values()), dtype=float)
    if (not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.0, rtol=1e-4, atol=0)):
        raise ValueError("mat mass fractions must be nonnegative and sum to one")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            _positive_scalar(mat.electron_density_per_mm3, "mat.electron_density_per_mm3")
            density = np.asarray(mat.density_params(), dtype=float)
            if (density.shape != (6,) or not np.all(np.isfinite(density))
                    or density[2] <= density[1] or np.any(density[3:] < 0)):
                raise ValueError("invalid material density-effect constants")
            fluct = mat.fluct_params()
            for name in ("I", "e0", "e1", "e2"):
                _positive_scalar(fluct[name], f"mat.fluct_params[{name}]")
            fractions = np.asarray([fluct["f1"], fluct["f2"]], dtype=float)
            if (not np.all(np.isfinite(fractions)) or np.any(fractions < 0)
                    or not np.isclose(fractions.sum(), 1.0)):
                raise ValueError("invalid material fluctuation fractions")
    except (FloatingPointError, OverflowError, KeyError) as exc:
        raise ValueError("invalid material loss constants") from exc


def _momentum(ke):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            p = np.sqrt(ke) * np.sqrt(ke + 2 * eloss.MMU)
            values = eloss.kinematics(p)
    except FloatingPointError as exc:
        raise ValueError("kinetic energy outside numerical range of loss model") from exc
    if (np.any(p < eloss.P_FLOOR) or any(not np.all(np.isfinite(v)) for v in values)
            or np.any(values[0] <= 0) or np.any(values[1] <= 0)):
        raise ValueError("kinetic energy cannot be represented without a loss-model floor")
    return p


def _loss_array(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.all(np.isfinite(value)) or np.any(value < 0):
        raise ValueError(f"{name} must be finite, nonnegative and match active lane shape")
    return value


def transport_segment(rng, mat, kinetic_MeV, length_mm, *, tcut_MeV,
                      stopping="mean-replace", max_step_mm=1.0, fractional_loss=0.05,
                      stop_ke_MeV=1.0, max_steps=10000):
    """Integrate one uniform segment per lane using an EXPERIMENTAL transport policy.

    Inputs: kinetic_MeV is a finite nonnegative 1-D array; length_mm is either a
    finite nonnegative scalar (broadcast) or a matching 1-D array. mat is a valid
    materials.Material; tcut_MeV, max_step_mm and stop_ke_MeV are finite positive
    scalars, fractional_loss is in (0, 1], and max_steps is a positive integer.
    All input/material/settings validation precedes sampling, including empty inputs.

    Returns {"slices": {...}, "final": {...}} containing independent 1-D arrays:
      slices: lane_index (integer original input index), s0_mm, s1_mm (local
        segment coordinates), ke_in_MeV, ke_out_MeV, sampled_loss_MeV (unaltered
        full proposed-step sampler result), applied_loss_MeV (actual KE removed),
        terminal ("none", "model_stop", or "exhausted"). Rows are iteration-major,
        ascending lane index within each iteration, after initial terminal rows.
      final: final_ke_MeV, traveled_mm, status ("transmitted", "model_stop", or
        "exhausted"), in original input order. Empty input gives empty arrays.

    Initial K <= stop_ke_MeV emits one zero-length row, with zero sampled/applied
    loss and residual KE retained; exact zero is "exhausted", otherwise "model_stop".
    Zero length with K > cutoff emits no rows and is transmitted without sampling.
    A processed endpoint at/below cutoff similarly retains all residual KE; no
    cutoff energy is dumped, no momentum/energy floor is applied. Terminal status
    takes precedence over transmission when the processed endpoint is the exit.

    Each iteration operates on all active lanes with step=min(remaining, max_step,
    fractional_loss*K/dedx); dedx==0 uses max_step without division by zero. Only
    overshoots (raw >= K) distinguish policies: "mean-replace" proposes dedx*step,
    "sampled" proposes raw. Other samples are unchanged. Proposal >= K exhausts
    KE at s_used=step*(K/proposal), retaining the original raw sample in the row.
    This linear within-step stopping location is a modeling assumption, NOT a
    demonstrated Geant4 range law. Applied loss is KE removed, not a statement
    that all restricted-loss energy is locally deposited.

    Nonfinite/negative model outputs raise ValueError. Unrepresentable advancement
    or the adaptive iteration cap raises RuntimeError, never a silent remainder
    or transmission. Numerical kinematics/model-domain failures raise ValueError.
    Errors after earlier steps may consume RNG state; there is no rollback.
    Repeating the same batch and seed is reproducible; scalar/batch RNG streams
    need not agree because active lanes share vectorized sampler calls. No time
    coordinate is computed. This API does not change production slab_loss.
    """
    ke = np.asarray(kinetic_MeV, dtype=float).copy()
    length = np.asarray(length_mm, dtype=float)
    if ke.ndim != 1 or not np.all(np.isfinite(ke)) or np.any(ke < 0):
        raise ValueError("kinetic_MeV must be a finite nonnegative 1-D array")
    if length.ndim > 1 or (length.ndim == 1 and length.shape != ke.shape):
        raise ValueError("length_mm must be scalar or match kinetic_MeV shape")
    if not np.all(np.isfinite(length)) or np.any(length < 0):
        raise ValueError("length_mm must be finite and nonnegative")
    length = np.broadcast_to(length, ke.shape).copy()
    tcut_MeV = _positive_scalar(tcut_MeV, "tcut_MeV")
    max_step_mm = _positive_scalar(max_step_mm, "max_step_mm")
    fractional_loss = _positive_scalar(fractional_loss, "fractional_loss")
    stop_ke_MeV = _positive_scalar(stop_ke_MeV, "stop_ke_MeV")
    if fractional_loss > 1:
        raise ValueError("fractional_loss must be in (0, 1]")
    if not isinstance(stopping, str) or stopping not in ("mean-replace", "sampled"):
        raise ValueError("stopping must be 'mean-replace' or 'sampled'")
    if isinstance(max_steps, (bool, np.bool_)) or not isinstance(max_steps, (int, np.integer)) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    _validate_material(mat)
    active = (ke > stop_ke_MeV) & (length > 0)
    _momentum(ke[active])
    s = np.zeros_like(ke)
    status = np.full(ke.shape, "transmitted", dtype="U11")
    status[ke <= stop_ke_MeV] = "model_stop"
    status[ke == 0] = "exhausted"
    columns = {name: [] for name in ("lane_index", "s0_mm", "s1_mm", "ke_in_MeV",
                                   "ke_out_MeV", "sampled_loss_MeV", "applied_loss_MeV", "terminal")}

    def record(lanes, s0, s1, before, after, raw, applied, terminal):
        for name, value in zip(columns, (lanes, s0, s1, before, after, raw, applied, terminal)):
            columns[name].append(value.copy())

    initial = np.flatnonzero(ke <= stop_ke_MeV)
    record(initial, s[initial], s[initial], ke[initial], ke[initial], s[initial], s[initial], status[initial])
    for _ in range(max_steps):
        lanes = np.flatnonzero(active)
        if not lanes.size:
            break
        before, s0 = ke[lanes], s[lanes]
        p = _momentum(before)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                dedx = _loss_array(eloss.dedx_restricted(mat, p, tcut_MeV), before.shape, "dedx")
        except FloatingPointError as exc:
            raise ValueError("restricted mean outside numerical model domain") from exc
        adaptive = np.full(before.shape, max_step_mm)
        with np.errstate(over="ignore", under="ignore"):
            np.divide(fractional_loss * before, dedx, out=adaptive, where=dedx > 0)
        step = np.minimum(np.minimum(length[lanes] - s0, max_step_mm), adaptive)
        if np.any(step <= 0) or np.any(s0 + step <= s0):
            raise RuntimeError("adaptive step cannot advance segment coordinate")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                raw = _loss_array(eloss.sample_loss(rng, mat, p, step, tcut_MeV), before.shape, "sampled loss")
        except FloatingPointError as exc:
            raise ValueError("sampler outside numerical model domain") from exc
        proposal = raw.copy()
        if stopping == "mean-replace":
            overshoot = raw >= before
            proposal[overshoot] = dedx[overshoot] * step[overshoot]
        proposal = _loss_array(proposal, before.shape, "applied proposal")
        exhausted = proposal >= before
        used = step.copy()
        used[exhausted] = step[exhausted] * (before[exhausted] / proposal[exhausted])
        s1 = np.minimum(s0 + used, length[lanes])
        if np.any(used <= 0) or np.any(s1 <= s0):
            raise RuntimeError("stopping location cannot advance segment coordinate")
        applied = np.minimum(proposal, before)
        after = before - applied
        if np.any((applied > 0) & (after == before)):
            raise RuntimeError("positive sampled debit cannot decrement kinetic energy at this precision")
        applied = before - after
        terminal = np.full(lanes.shape, "none", dtype="U10")
        terminal[after <= stop_ke_MeV] = "model_stop"
        terminal[after == 0] = "exhausted"
        record(lanes, s0, s1, before, after, raw, applied, terminal)
        ke[lanes], s[lanes] = after, s1
        stopped = terminal != "none"
        status[lanes[stopped]] = terminal[stopped]
        active[lanes] = ~stopped & (s1 < length[lanes])
    if np.any(active):
        raise RuntimeError(f"segment transport exceeded max_steps={max_steps}")
    slices = {name: np.concatenate(values) for name, values in columns.items()}
    return {"slices": slices, "final": {"final_ke_MeV": ke, "traveled_mm": s, "status": status}}
