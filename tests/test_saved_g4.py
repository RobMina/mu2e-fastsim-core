"""Synthetic saved-G4 schema, identity, accounting and legacy-comparison regressions."""
from contextlib import contextmanager

import numpy as np
import pytest

from mu2efastsim import saved_g4


BIG = 2**53
SCHEMA = {
    "events": {
        "int": "schema_version n_collections n_missing",
        "bool": "complete_boundary_records",
    },
    "collections": {
        "int": "product_id count", "bool": "present", "str": "kind tag",
    },
    "particles": {
        "int": "sim_product_id sim_key parent_product_id parent_key gen_product_id gen_key "
               "pdg stage start_volume end_volume start_status end_status creation stopping n_steps float_level",
        "bool": "parent_available gen_available end_defined truncated self_parent",
        "float": "t0 t1 length tau0 tau1 e0 e1 end_ke pre_last_ke start_excitation",
        "vector": "x0 x1 p0 p1",
    },
    "daughters": {
        "int": "sim_product_id sim_key daughter_index daughter_product_id daughter_key",
        "bool": "daughter_available",
    },
    "generators": {
        "int": "pdg gen_id", "float": "t0 e0 tau0", "vector": "x0 p0",
    },
    "crv_steps": {
        "int": "sim_product_id sim_key bar",
        "float": "t0 t1 visible end_momentum_magnitude length", "vector": "x0 x1 p0",
    },
    "step_points": {
        "int": "sim_product_id sim_key volume process",
        "float": "t0 tau0 length edep nonion visible", "vector": "x0 x1 p0 p1",
    },
    "trajectories": {"int": "sim_product_id sim_key n_points"},
    "trajectory_points": {
        "int": "sim_product_id sim_key point_index", "float": "t0 ke", "vector": "x0",
    },
    "surface_steps": {
        "int": "sim_product_id sim_key surface_id surface_index",
        "float": "t0 edep", "vector": "x0 x1 p0",
    },
    "status": {
        "int": "status n_tracks killed_step_limit killed_field", "bool": "overflow",
    },
}


def make_table(name, size):
    """Allocate every branch in SavedG4TableDumper's schema, including empty vectors."""
    table = {key: np.zeros(size, dtype=np.int64) for key in saved_g4.EVENT}
    if name not in ("events", "collections"):
        table.update({key: np.zeros(size, dtype=np.int64) for key in ("product_id", "index")})
        table["tag"] = np.full(size, name, dtype=object)
    dtypes = {"int": np.int64, "float": np.float64, "bool": np.bool_, "str": object}
    for kind, names in SCHEMA[name].items():
        for key in names.split():
            table[key] = (np.zeros((size, 3), dtype=np.float64) if kind == "vector"
                          else np.zeros(size, dtype=dtypes[kind]))
    return table


def take(table, indices):
    return {key: value[indices].copy() for key, value in table.items()}


@pytest.fixture
def tables():
    """Two events, all configured products, adjacent large keys and an empty trajectory."""
    result = {name: make_table(name, 2 if name == "events" else 14 if name == "collections" else 3)
              for name in saved_g4.TABLES}
    events = result["events"]
    events["run"][:] = [1, 2]
    events["subrun"][:] = [3, 4]
    events["event"][:] = 17
    events["schema_version"][:] = 1
    events["n_collections"][:] = len(saved_g4.KINDS)
    collections = result["collections"]
    for key in saved_g4.EVENT:
        collections[key][:] = np.repeat(events[key], len(saved_g4.KINDS))
    collections["kind"][:] = np.tile(saved_g4.KINDS, 2)
    collections["tag"][:] = collections["kind"]
    collections["product_id"][:] = np.tile(np.arange(10, 17), 2)
    collections["present"][:] = True
    collections["count"][:] = np.repeat([2, 1], len(saved_g4.KINDS))
    products = dict(zip(saved_g4.KINDS, range(10, 17)))
    products.update(daughters=10, trajectory_points=14)
    for name, table in result.items():
        if name in ("events", "collections"):
            continue
        for key in saved_g4.EVENT:
            table[key][:] = events[key][[0, 0, 1]]
        table["product_id"][:] = products[name]
        table["index"][:] = [0, 1, 0]
        if "sim_key" in table:
            table["sim_product_id"][:] = 10
            table["sim_key"][:] = np.array([BIG, BIG + 1, BIG + 1], dtype=np.int64)
    particles = result["particles"]
    particles["pdg"][:] = [13, 11, 13]
    particles["end_defined"][:] = True
    particles["parent_product_id"][:] = [0, 10, 0]
    particles["parent_key"][:] = np.array([-1, BIG, -1], dtype=np.int64)
    particles["parent_available"][:] = [False, True, False]
    particles["gen_product_id"][:] = [0, 11, 0]
    particles["gen_key"][:] = [-1, 0, -1]
    particles["gen_available"][:] = [False, True, False]
    result["daughters"]["daughter_product_id"][:] = 10
    result["daughters"]["daughter_key"][:] = np.array([BIG + 1, BIG, BIG + 1], dtype=np.int64)
    result["daughters"]["daughter_available"][:] = True
    result["trajectories"]["n_points"][:] = [2, 0, 1]
    points = result["trajectory_points"]
    points["index"][:] = 0
    points["point_index"][:] = [0, 1, 0]
    points["sim_key"][:] = np.array([BIG, BIG, BIG + 1], dtype=np.int64)
    points["tag"][:] = "trajectories"
    result["daughters"]["tag"][:] = "particles"
    steps = result["crv_steps"]
    steps["bar"][:] = [41, 42, 43]
    for offset, key in enumerate(SCHEMA["crv_steps"]["float"].split()):
        steps[key][:] = np.arange(3) + 0.123456789 + offset
    for offset, key in enumerate(SCHEMA["crv_steps"]["vector"].split()):
        steps[key][:] = np.arange(9).reshape(3, 3) + 0.123456789 + offset * 10
    return result


def legacy_steps(tables, with_product=True):
    steps = tables["crv_steps"]
    result = {key: steps[key].copy() for key in saved_g4.EVENT}
    for old, new in {"bar": "bar", "sp": "sim_key", "edep": "visible", "t0": "t0", "t1": "t1",
                     "p1": "end_momentum_magnitude", "len": "length"}.items():
        result[old] = steps[new].copy()
    for i, axis in enumerate("xyz"):
        result[f"{axis}0"] = steps["x0"][:, i].copy()
        result[f"{axis}1"] = steps["x1"][:, i].copy()
        result[f"p{axis}0"] = steps["p0"][:, i].copy()
    if with_product:
        result["sp_product_id"] = steps["sim_product_id"].copy()
    return {key: value.astype(np.float32) if value.dtype.kind == "f" else value
            for key, value in result.items()}


def test_read_requests_all_tables_without_explicit_cycles(monkeypatch, tables):
    requested = []

    class Tree:
        def __init__(self, name):
            self.name = name

        def arrays(self, *, library):
            assert library == "np"
            return tables[self.name]

    class Root:
        def __getitem__(self, key):
            requested.append(key)
            assert key.startswith("custom/")
            assert ";" not in key
            return Tree(key.split("/")[1])

    @contextmanager
    def open_root(path):
        assert path == "synthetic.root"
        yield Root()

    monkeypatch.setattr(saved_g4.uproot, "open", open_root)
    actual = saved_g4.read("synthetic.root", directory="custom")
    assert requested == [f"custom/{name}" for name in saved_g4.TABLES]
    assert all(actual[name] is tables[name] for name in saved_g4.TABLES)


def test_census_and_audit_valid_full_schema(tables):
    assert len(saved_g4.census(tables)) == 14
    report = saved_g4.audit(tables)
    assert report["events"] == 2
    assert report["rows"] == {name: len(table["event"]) for name, table in tables.items()}
    assert report["missing_collections"] == report["empty_collections"] == 0
    assert report["complete_component_bank"] is False
    assert report["links"] == {
        "parents": {"links": 1, "outside_export": 0},
        "daughters": {"links": 3, "outside_export": 0},
        "generators": {"links": 1, "outside_export": 0},
    }
    assert report["retained_particle_coverage"] == [
        {"pdg": 11, "count": 1, "trajectory": 1, "crv": 1, "raw_step": 1, "surface": 1,
         "crv_without_trajectory": 0},
        {"pdg": 13, "count": 2, "trajectory": 2, "crv": 2, "raw_step": 2, "surface": 2,
         "crv_without_trajectory": 0},
    ]


def test_empty_export_has_zero_census_and_no_completeness_claim(tables):
    empty = {name: take(table, slice(0, 0)) for name, table in tables.items()}
    report = saved_g4.audit(empty)
    assert report["events"] == 0
    assert report["missing_collections"] == report["empty_collections"] == 0
    assert report["retained_particle_coverage"] == []
    assert report["complete_component_bank"] is False


def test_census_distinguishes_present_empty_from_missing(tables):
    col = tables["collections"]
    empty = col["kind"] == "surface_steps"
    missing = col["kind"] == "step_points"
    col["count"][empty] = 0
    col["count"][missing] = -1
    col["present"][missing] = False
    col["product_id"][missing] = 0
    tables["events"]["n_missing"][:] = 1
    for name in ("surface_steps", "step_points"):
        tables[name] = take(tables[name], slice(0, 0))
    report = saved_g4.audit(tables)
    assert report["empty_collections"] == 2
    assert report["missing_collections"] == 2


@pytest.mark.parametrize("column,value,message", [
    ("schema_version", 2, "unsupported saved-G4 schema"),
    ("complete_boundary_records", True, "cannot certify complete boundary"),
    ("n_collections", 6, "event n_collections mismatch"),
    ("n_missing", 1, "event n_missing mismatch"),
])
def test_event_certification_and_counters(tables, column, value, message):
    tables["events"][column][0] = value
    with pytest.raises(ValueError, match=message):
        saved_g4.census(tables)


@pytest.mark.parametrize("name,message", [
    ("events", "duplicate event identity"),
    ("collections", "duplicate collection identity"),
    ("crv_steps", "duplicate crv_steps identity"),
    ("trajectory_points", "duplicate trajectory point identity"),
    ("daughters", "duplicate daughter identity"),
])
def test_duplicate_table_identities(tables, name, message):
    tables[name] = take(tables[name], np.r_[np.arange(len(tables[name]["event"])), 0])
    with pytest.raises(ValueError, match=message):
        saved_g4.census(tables)


def test_duplicate_particle_identity_with_distinct_row_indices(tables):
    tables["particles"]["sim_key"][1] = BIG
    with pytest.raises(ValueError, match="duplicate particle identity"):
        saved_g4.census(tables)


def test_collection_alias_is_not_a_second_product(tables):
    col = tables["collections"]
    tables["collections"] = take(col, np.r_[np.arange(14), 0])
    tables["collections"]["tag"][-1] = "different-label"
    tables["events"]["n_collections"][0] += 1
    with pytest.raises(ValueError, match="aliased collection tags"):
        saved_g4.census(tables)


@pytest.mark.parametrize("present,count,product,message", [
    (False, 0, 0, "missing product must have count -1"),
    (True, -1, 10, "present product has negative count"),
    (False, -1, 10, "product ID/presence disagreement"),
    (True, 2, 0, "product ID/presence disagreement"),
    (True, 2, -1, "product ID/presence disagreement"),
])
def test_invalid_collection_presence(tables, present, count, product, message):
    col = tables["collections"]
    col["present"][0], col["count"][0], col["product_id"][0] = present, count, product
    with pytest.raises(ValueError, match=message):
        saved_g4.census(tables)


@pytest.mark.parametrize("kind", saved_g4.KINDS)
def test_missing_product_row_fails_census(tables, kind):
    tables[kind] = take(tables[kind], [0, 2])
    with pytest.raises(ValueError, match=f"{kind} census mismatch"):
        saved_g4.census(tables)


def test_collection_row_missing_from_export(tables):
    col = tables["collections"]
    tables["collections"] = take(col, col["kind"] != "crv_steps")
    tables["events"]["n_collections"][:] -= 1
    with pytest.raises(ValueError, match="unmatched crv_steps collection"):
        saved_g4.census(tables)


@pytest.mark.parametrize("column,value,message", [
    ("product_id", 999, "unmatched trajectory point header"),
    ("index", 1, "unmatched trajectory point header"),
    ("sim_product_id", 999, "unmatched trajectory point header"),
    ("sim_key", BIG + 1, "unmatched trajectory point header"),
    ("point_index", -1, "point index out of range"),
    ("point_index", 2, "point index out of range"),
])
def test_trajectory_point_header_identity_and_bounds(tables, column, value, message):
    tables["trajectory_points"][column][0] = value
    with pytest.raises(ValueError, match=message):
        saved_g4.census(tables)


def test_missing_trajectory_point_fails_count(tables):
    tables["trajectory_points"] = take(tables["trajectory_points"], [0, 2])
    with pytest.raises(ValueError, match="trajectory point census mismatch"):
        saved_g4.census(tables)


def test_empty_trajectory_header_cannot_claim_points(tables):
    tables["trajectories"]["n_points"][1] = 1
    with pytest.raises(ValueError, match="trajectory point census mismatch"):
        saved_g4.census(tables)


@pytest.mark.parametrize("name,column", [
    ("events", "event"), ("collections", "product_id"),
    ("particles", "sim_key"), ("particles", "parent_key"),
    ("trajectory_points", "point_index"), ("daughters", "daughter_index"),
])
def test_noninteger_identity_columns_are_rejected(tables, name, column):
    tables[name][column] = tables[name][column].astype(np.float64)
    with pytest.raises(ValueError, match="noninteger"):
        saved_g4.census(tables)


def test_negative_retained_particle_key_is_invalid_even_when_references_agree(tables):
    for table in tables.values():
        if "sim_key" in table:
            table["sim_key"][table["sim_key"] == BIG] = -1
    with pytest.raises(ValueError):
        saved_g4.census(tables)


@pytest.mark.parametrize("kind", ["crv_steps", "step_points", "trajectories", "surface_steps", "daughters"])
def test_response_particle_outside_export_fails(tables, kind):
    tables[kind]["sim_key"][0] = BIG + 7
    with pytest.raises(ValueError, match=f"unmatched {kind} particle"):
        saved_g4.census(tables)


def test_large_integer_identity_does_not_round_into_an_existing_particle(tables):
    assert tables["particles"]["sim_key"].dtype == np.dtype(np.int64)
    saved_g4.census(tables)
    tables["particles"]["sim_key"][1] = BIG + 2
    with pytest.raises(ValueError, match="unmatched crv_steps particle"):
        saved_g4.census(tables)


def test_link_closure_counts_outside_export_without_rejecting_retained_links(tables):
    particles = tables["particles"]
    particles["parent_product_id"][:] = [10, 10, 10]
    particles["parent_key"][:] = np.array([BIG + 4, BIG, BIG], dtype=np.int64)
    particles["gen_product_id"][:] = [11, 11, 999]
    particles["gen_key"][:] = [99, 0, 0]
    tables["daughters"]["daughter_key"][:] = np.array([BIG + 1, BIG + 4, BIG], dtype=np.int64)
    report = saved_g4.audit(tables)
    assert report["links"] == {name: {"links": 3, "outside_export": 2}
                               for name in ("parents", "daughters", "generators")}


@pytest.mark.parametrize("product,key", [(0, BIG), (10, -1)])
def test_null_link_identity_must_be_consistent(tables, product, key):
    tables["particles"]["parent_product_id"][0] = product
    tables["particles"]["parent_key"][0] = key
    with pytest.raises(ValueError, match="null pointer identity mismatch"):
        saved_g4.audit(tables)


def test_audit_reports_response_without_trajectory_and_integrity_counters(tables):
    tables["trajectories"] = take(tables["trajectories"], [0, 2])
    col = tables["collections"]
    col["count"][(col["kind"] == "trajectories") & (col["run"] == 1)] = 1
    tables["particles"]["truncated"][0] = True
    tables["particles"]["end_defined"][1] = False
    status = tables["status"]
    status["status"][:] = [0, 2, 3]
    status["n_tracks"][:] = [4, 5, 6]
    status["overflow"][:] = [False, True, False]
    status["killed_step_limit"][:] = [0, 2, 1]
    status["killed_field"][:] = [1, 0, 1]
    report = saved_g4.audit(tables)
    electron = next(row for row in report["retained_particle_coverage"] if row["pdg"] == 11)
    assert electron["trajectory"] == 0
    assert electron["crv_without_trajectory"] == 1
    assert report["truncated_particles"] == report["undefined_particle_ends"] == 1
    assert report["g4_status"] == {"rows": 3, "nonzero": 2, "tracks": 15, "overflow_events": 1,
                                    "killed_step_limit": 3, "killed_field": 2}


@pytest.mark.parametrize("with_product", [False, True])
def test_compare_crv_equal_at_legacy_precision(tables, with_product):
    legacy = legacy_steps(tables, with_product)
    assert legacy["edep"][0] != tables["crv_steps"]["visible"][0]
    report = saved_g4.compare_crv(tables, legacy)
    assert report["rows"] == 3
    assert report["particle_ProductID_checked"] is with_product
    assert set(report["fields_equal_at_legacy_precision"]) == set(legacy) - set(saved_g4.EVENT)


@pytest.mark.parametrize("field", [
    "bar", "sp", "sp_product_id", "edep", "t0", "t1", "p1", "len",
    "x0", "y0", "z0", "x1", "y1", "z1", "px0", "py0", "pz0",
])
def test_compare_crv_rejects_each_common_field_mismatch(tables, field):
    legacy = legacy_steps(tables)
    legacy[field][0] += 1
    with pytest.raises(ValueError, match=f"field mismatches:.*'{field}':"):
        saved_g4.compare_crv(tables, legacy)


def test_compare_crv_selects_exact_exported_events_and_reorders_event_blocks(tables):
    legacy = take(legacy_steps(tables), [2, 0, 0, 1, 0])
    legacy["run"][1] = 999
    legacy["subrun"][4] = 999
    legacy["edep"][[1, 4]] = -999
    tables["crv_steps"] = take(tables["crv_steps"], [1, 2, 0])
    assert saved_g4.compare_crv(tables, legacy)["rows"] == 3


@pytest.mark.parametrize("indices", [[0, 2], [0, 1, 1, 2]])
def test_compare_crv_rejects_missing_or_extra_selected_rows(tables, indices):
    with pytest.raises(ValueError, match="legacy CRV row count mismatch"):
        saved_g4.compare_crv(tables, take(legacy_steps(tables), indices))


@pytest.mark.parametrize("index", [-1, 2])
def test_collection_row_index_out_of_range(tables, index):
    tables["crv_steps"]["index"][0] = index
    with pytest.raises(ValueError, match="crv_steps index out of range"):
        saved_g4.census(tables)


def test_particle_collection_and_reference_product_must_agree(tables):
    tables["particles"]["sim_product_id"][0] = 99
    with pytest.raises(ValueError, match="particle collection/reference ProductID mismatch"):
        saved_g4.census(tables)


def test_int64_product_ids_above_float_exactness_limit(tables):
    for table in tables.values():
        for key in ("product_id", "sim_product_id", "parent_product_id", "gen_product_id", "daughter_product_id"):
            if key in table:
                table[key][table[key] > 0] += BIG + 1
    report = saved_g4.audit(tables)
    assert all(link["outside_export"] == 0 for link in report["links"].values())
    assert saved_g4.compare_crv(tables, legacy_steps(tables))["particle_ProductID_checked"] is True


def test_compare_crv_empty_export_ignores_all_legacy_rows(tables):
    legacy = legacy_steps(tables)
    empty = {name: take(table, slice(0, 0)) for name, table in tables.items()}
    assert saved_g4.compare_crv(empty, legacy)["rows"] == 0


def test_compare_crv_cannot_silently_compare_multiple_collections(tables):
    tables["crv_steps"]["index"][1] = 0
    tables["crv_steps"]["product_id"][1] = 999
    with pytest.raises(ValueError, match="duplicate single CRV collection identity"):
        saved_g4.compare_crv(tables, legacy_steps(tables))


def test_legacy_comparison_preserves_matching_nan_but_audit_reports_it(tables):
    tables["crv_steps"]["t1"][0] = np.nan
    legacy = legacy_steps(tables)
    assert saved_g4.compare_crv(tables, legacy)["rows"] == 3
    assert saved_g4.audit(tables)["nonfinite_fields"]["crv_steps"] == {"t1": 1}
    legacy["t1"][0] = 1.
    with pytest.raises(ValueError, match="field mismatches"):
        saved_g4.compare_crv(tables, legacy)


def test_legacy_comparison_never_wraps_particle_integer_key(tables):
    legacy = legacy_steps(tables)
    legacy["sp"] = legacy["sp"].astype(np.int32)
    with pytest.raises(ValueError, match="legacy sp integer overflow"):
        saved_g4.compare_crv(tables, legacy)
