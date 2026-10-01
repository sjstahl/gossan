from pathlib import Path

import pytest

from gossan.harvest.bc.minfile import MinfileConnector

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def aoi_file() -> Path:
    return FIXTURES / "aoi_bc.geojson"


@pytest.fixture
def wfs_calls(monkeypatch) -> list[str]:
    """Replace the MINFILE network call with the committed fixture; returns requested URLs."""
    calls: list[str] = []

    def fake_get(self, url: str) -> bytes:
        calls.append(url)
        return (FIXTURES / "minfile_wfs.geojson").read_bytes()

    monkeypatch.setattr(MinfileConnector, "_get", fake_get)
    return calls
