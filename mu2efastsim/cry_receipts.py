"""Dependency-free validation of emitted independent native-CRY shard exposures."""
import math


GENERATOR = "native-cry-1.7"
DEVICE_GENERATOR = "jax-cry-1.7"
EXTRACTED_V04_GDML_SHA256 = "5b357107ac740de5df770066b6428c1a58ec61e0349380219ab08dd9d1ace3d9"


def _integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def _positive_finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value) and value > 0
    except OverflowError:
        return False


def merge_receipts(records):
    """Refuse mixed/duplicate native shards; legacy all-table records retain their contract."""
    native = ["native_generator" in r for r in records]
    if not any(native):
        return {}
    if not all(native):
        raise ValueError("native generator merge: cannot mix native and table shards")
    receipts, seeds, ranges, signature = [], set(), {}, None
    for record in records:
        receipt = record["native_generator"]
        if not isinstance(receipt, dict) or receipt.get("generator") not in (GENERATOR, DEVICE_GENERATOR):
            raise ValueError("native generator merge: invalid generator receipt")
        if receipt["generator"] == DEVICE_GENERATOR:
            from .cry_constants import SEA_LEVEL_SHA256
            sampling = receipt.get("sampling")
            if (not isinstance(sampling, dict) or sampling.get("table_sha256") != SEA_LEVEL_SHA256
                    or sampling.get("subbox_m") != 300.0 or sampling.get("date") != "2021-06-21"
                    or sampling.get("latitude") != 41.8 or not _positive_finite(sampling.get("lifetime_s"))
                    or not isinstance(receipt.get("rng"), str) or not receipt["rng"].startswith("jax threefry2x32")
                    or not isinstance(receipt.get("device"), str)
                    or not receipt["device"].startswith(("cuda:", "gpu:"))
                    or not _integer(receipt.get("particle_capacity"), 1, 2**31-1)
                    or not _integer(receipt.get("selected_capacity"), 1, 2**31-1)):
                raise ValueError("native generator merge: invalid device sampling provenance")
        count, seed, live = (receipt.get(k) for k in ("accepted_events", "seed", "live_seconds"))
        if (not _integer(count, 1, 2**20 - 1) or not _integer(seed, 0, 2**64 - 1)
                or not _positive_finite(live)
                or not _integer(record.get("generated_events"), count, count) or record.get("live_seconds") != live):
            raise ValueError("native generator merge: invalid count, seed or live exposure")
        current = {k: receipt.get(k) for k in ("generator", "cosmic_config", "cosmic_time_offset",
                                              "stock_overenergy_accumulation", "raw_batch_size", "sampling", "rng",
                                              "particle_capacity", "selected_capacity")}
        if (not isinstance(current["cosmic_config"], dict) or not current["cosmic_config"]
                or not isinstance(current["cosmic_time_offset"], str)
                or current["stock_overenergy_accumulation"] is not True
                or not _integer(current["raw_batch_size"], 1, 100000)):
            raise ValueError("native generator merge: incomplete configuration provenance")
        if signature is not None and current != signature:
            raise ValueError("native generator merge: incompatible generator configurations")
        signature = current
        event_range = receipt.get("event_range")
        if not isinstance(event_range, dict):
            raise ValueError("native generator merge: missing event range")
        run, subrun, first, last = (event_range.get(k) for k in ("run", "subrun", "first_event", "last_event"))
        if (not _integer(run, 0, 2**31 - 1) or not _integer(subrun, 0, 2**28 - 1)
                or not _integer(first, 1, 2**20 - 1) or not _integer(last, first, 2**20 - 1)
                or last - first + 1 != count):
            raise ValueError("native generator merge: invalid packed event range")
        if seed in seeds:
            raise ValueError("native generator merge: repeated fresh-source seed")
        for a, b in ranges.get(subrun, []):
            if first <= b and a <= last:
                raise ValueError("native generator merge: overlapping event identifiers")
        ranges.setdefault(subrun, []).append((first, last))
        seeds.add(seed)
        receipts.append(receipt)
    return {"generator": signature["generator"], "live_seconds": math.fsum(r["live_seconds"] for r in receipts),
            "native_generators": receipts}
