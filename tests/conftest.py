from pathlib import Path

import pytest

from gossan.harvest.bc.aris_pdf import ArisPdfDownloader
from gossan.harvest.bc.wfs import WfsConnector

FIXTURES = Path(__file__).parent / "fixtures"

WFS_FIXTURES = {
    "pub:WHSE_MINERAL_TENURE.MINFIL_MINERAL_FILE": "minfile_wfs.geojson",
    "pub:WHSE_MINERAL_TENURE.MTA_ACQUIRED_TENURE_SVW": "mto_tenure_wfs.geojson",
    "pub:WHSE_MINERAL_TENURE.ARIS_MINERAL_REPORTS": "aris_index_wfs.geojson",
    "pub:WHSE_MINERAL_TENURE.GEOL_REG_GEOCHMCL_SURV_PNT_SP": "rgs_wfs.geojson",
}


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def aoi_file() -> Path:
    return FIXTURES / "aoi_bc.geojson"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


class FakeWfs:
    """Stands in for the network: serves fixture files by requested layer."""

    def __init__(self) -> None:
        self.responses: dict[str, str] = dict(WFS_FIXTURES)  # layer -> fixture file name
        self.calls: list[str] = []  # requested URLs


@pytest.fixture
def wfs(monkeypatch) -> FakeWfs:
    """Replace the WFS network call with committed fixtures; no live requests in tests."""
    fake = FakeWfs()

    def fake_get(self, url: str) -> bytes:
        fake.calls.append(url)
        return (FIXTURES / fake.responses[self.type_name]).read_bytes()

    monkeypatch.setattr(WfsConnector, "_get", fake_get)
    return fake


class FakeAris:
    """Stands in for the ARIS PDF server."""

    def __init__(self) -> None:
        self.missing: set[str] = set()  # URLs that answer 404
        self.not_pdf: set[str] = set()  # URLs that answer 200 with an HTML page
        self.calls: list[str] = []


@pytest.fixture
def aris_pdfs(monkeypatch) -> FakeAris:
    """Replace the PDF download with a tiny fixture PDF; also removes the polite pause."""
    fake = FakeAris()

    def fake_download(self, url: str, part: Path) -> None:
        fake.calls.append(url)
        if url in fake.missing:
            raise OSError(f"HTTP 404 for {url}")
        if url in fake.not_pdf:
            part.write_bytes(b"<html>Sign in</html>")
        else:
            part.write_bytes((FIXTURES / "dummy_report.pdf").read_bytes())

    monkeypatch.setattr(ArisPdfDownloader, "_download", fake_download)
    monkeypatch.setattr("gossan.harvest.bc.aris_pdf.PAUSE_SECONDS", 0.0)
    return fake
