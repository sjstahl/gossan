"""Saving downloads under raw/ without keeping identical copies."""

from datetime import UTC, datetime, timedelta

from gossan.core.rawstore import sha256_of, store_raw

T0 = datetime(2026, 10, 11, 8, 0, 0, tzinfo=UTC)


def test_first_download_is_written_with_a_timestamped_name(tmp_path):
    stored = store_raw(tmp_path / "raw" / "minfile", "minfile", "geojson", b"{}", T0)
    assert not stored.unchanged
    assert stored.path.name == "minfile_20261011T080000000000Z.geojson"
    assert stored.path.read_bytes() == b"{}"
    assert stored.sha256 == sha256_of(stored.path)


def test_identical_download_reuses_the_file_and_changed_one_adds_a_file(tmp_path):
    folder = tmp_path / "raw"
    first = store_raw(folder, "d", "tif", b"AAAA", T0)
    again = store_raw(folder, "d", "tif", b"AAAA", T0 + timedelta(days=1))
    assert again.unchanged and again.path == first.path and len(list(folder.iterdir())) == 1

    changed = store_raw(folder, "d", "tif", b"AAAB", T0 + timedelta(days=2))  # same size, different bytes
    assert not changed.unchanged and changed.path != first.path
    assert first.path.read_bytes() == b"AAAA"  # the original is untouched
    # Going back to the first content reuses the first file, not the newest.
    back = store_raw(folder, "d", "tif", b"AAAA", T0 + timedelta(days=3))
    assert back.unchanged and back.path == first.path
    assert len(list(folder.iterdir())) == 2


def test_other_stems_and_extensions_do_not_match(tmp_path):
    a = store_raw(tmp_path, "dem", "tif", b"same", T0)
    b = store_raw(tmp_path, "mag", "tif", b"same", T0)  # different product, same bytes
    c = store_raw(tmp_path, "dem", "geojson", b"same", T0)
    assert not b.unchanged and not c.unchanged and len({a.path, b.path, c.path}) == 3


# --- WFS responses whose feature ids change on every request --------------------------------


def _feature(fid: str, object_id: int | None = 7) -> dict:
    props = {"NAME": "x"} if object_id is None else {"NAME": "x", "OBJECTID": object_id}
    return {"type": "Feature", "id": fid, "geometry": None, "properties": props}


def test_session_ids_become_stable_and_other_ids_are_left_alone():
    from gossan.harvest.bc_provincial.wfs import normalize_session_ids

    features = [
        _feature("WHSE_X.MTA_TENURE.fid-509cbb71_1a127e60453_-343b", 7),
        _feature("WHSE_X.MINFIL.61942497", 61942497),  # already a stable id
        _feature("WHSE_X.NOOID.fid-abc_1", None),  # nothing stable to use: kept
    ]
    assert normalize_session_ids(features) == 1
    assert [f["id"] for f in features] == ["WHSE_X.MTA_TENURE.7", "WHSE_X.MINFIL.61942497", "WHSE_X.NOOID.fid-abc_1"]


def test_unchanged_wfs_data_dedupes_even_though_the_server_invents_new_ids(aoi_file, monkeypatch):
    import json
    from pathlib import Path

    from typer.testing import CliRunner

    from gossan.cli import app
    from gossan.core.project import Project
    from gossan.harvest.bc_provincial.wfs import WfsConnector

    runner = CliRunner()
    calls = {"n": 0}

    def fake_get(self, url):
        calls["n"] += 1  # a new session id each request, same data
        feats = [
            {**_feature(f"L.{i}.fid-deadbeef_{calls['n']:04d}_-{i:x}", i), "geometry": {"type": "Point", "coordinates": [-120.3, 50.71]}}
            for i in (1, 2)
        ]
        return json.dumps({"type": "FeatureCollection", "features": feats}).encode()

    monkeypatch.setattr(WfsConnector, "_get", fake_get)
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)]).exit_code == 0
    project = Project(Path("demo"))
    for _ in range(2):
        assert runner.invoke(app, ["harvest", "demo", "--source", "parks_protected"]).exit_code == 0
    assert calls["n"] == 2 and len(list((project.raw_dir / "parks_protected").glob("*.geojson"))) == 1
    fetches = [r for r in project.provenance.read() if r.operation == "fetch"]
    assert [f.parameters["unchanged"] for f in fetches] == [False, True]
    assert fetches[0].parameters["normalized_session_ids"] == 2
