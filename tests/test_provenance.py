import json

import pytest
from pydantic import ValidationError

from gossan import __version__
from gossan.core.provenance import ProvenanceLog, ProvenanceRecord

FIELDS = {
    "dataset": "minfile",
    "source": "BC",
    "identifier": "layer",
    "url": "https://example.test",
    "license": "OGL-BC",
    "operation": "fetch",
    "parameters": {"n": 1},
}


def test_append_and_read_roundtrip(tmp_path):
    log = ProvenanceLog(tmp_path)
    log.log(**FIELDS)
    log.log(**{**FIELDS, "operation": "write_layer"})

    lines = (tmp_path / "provenance.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert set(first) == {
        "dataset", "source", "identifier", "url", "retrieved_at",
        "license", "operation", "parameters", "gossan_version",
    }  # fmt: skip
    assert first["gossan_version"] == __version__
    assert [r.operation for r in log.read()] == ["fetch", "write_layer"]


def test_schema_is_enforced(tmp_path):
    with pytest.raises(ValidationError):
        ProvenanceRecord(**{**FIELDS, "unexpected": 1})
    incomplete = {k: v for k, v in FIELDS.items() if k != "license"}
    with pytest.raises(ValidationError):
        ProvenanceLog(tmp_path).log(**incomplete)
    assert not (tmp_path / "provenance.jsonl").exists()


def test_read_missing_log_is_empty(tmp_path):
    assert ProvenanceLog(tmp_path).read() == []
