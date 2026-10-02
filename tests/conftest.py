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


@pytest.fixture
def seed_leads_layers(fixtures_dir):
    """Write the mocked tenure and ARIS fixtures into a project's data.gpkg."""
    import geopandas as gpd

    def seed(project) -> None:
        for fixture, layer in (("leads_tenure", "mto_tenure"), ("leads_aris", "aris_index")):
            gdf = gpd.read_file(fixtures_dir / f"{fixture}.geojson").to_crs(project.config.crs)
            gdf.to_file(project.data_gpkg, layer=layer, driver="GPKG", engine="pyogrio")

    return seed


# --- OCR and Claude: never real in tests ---------------------------------------------------


@pytest.fixture(autouse=True)
def no_real_ocr_or_claude(monkeypatch):
    """Block EasyOCR (model downloads) and the Anthropic client (network) unless a test fakes them."""
    import sys

    from gossan.digitize import claude_vision, ocr

    monkeypatch.setitem(sys.modules, "easyocr", None)  # `import easyocr` raises ImportError
    monkeypatch.setattr(ocr, "_reader", None)

    def forbidden(*args, **kwargs):
        raise AssertionError("a test tried to create a live Anthropic client")

    monkeypatch.setattr(claude_vision.anthropic, "Anthropic", forbidden)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOSSAN_CLAUDE_MODEL", raising=False)


class FakeEasyOcr:
    """A stand-in `easyocr` module; records how it was used."""

    def __init__(self) -> None:
        self.reader_args: list[tuple] = []
        self.images: list[str] = []  # file names OCR was asked to read

        outer = self

        class Reader:
            def __init__(self, langs, gpu=True):
                outer.reader_args.append((tuple(langs), gpu))

            def readtext(self, image, detail=1, paragraph=False):
                name = Path(image).name
                outer.images.append(name)
                assert detail == 0 and paragraph  # plain text out, lines merged into paragraphs
                return [f"Scanned text from {name}", "Au 12 ppb Cu 40 ppm"]

        self.Reader = Reader


@pytest.fixture
def fake_easyocr(monkeypatch) -> FakeEasyOcr:
    import sys

    fake = FakeEasyOcr()
    monkeypatch.setitem(sys.modules, "easyocr", fake)
    return fake


class FakeClaude:
    """A stand-in Anthropic client: `messages.parse` returns queued PageTable results."""

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.queue: list = []  # PageTable instances or exceptions, used in order
        self.default = None
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        item = self.queue.pop(0) if self.queue else self.default
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(
            parsed_output=item,
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1500, output_tokens=300),
        )


@pytest.fixture
def fake_claude(monkeypatch) -> FakeClaude:
    from gossan.digitize import claude_vision
    from gossan.digitize.claude_vision import Attribute, PageTable, TableRow

    fake = FakeClaude()
    fake.default = PageTable(
        table_type="assay",
        depth_unit="m",
        notes="Bottom rows smudged.",
        rows=[
            TableRow(
                hole_id="R-1", from_m=0.0, to_m=1.5,
                attributes=[Attribute(name="Au (ppb)", value="12"), Attribute(name="Lithology", value="andesite")],
            ),
            TableRow(
                hole_id="R-1", from_m=1.5, to_m=3.0,
                attributes=[Attribute(name="Au (ppb)", value="<5"), Attribute(name="Cu (ppm)", value="40")],
            ),
        ],
    )  # fmt: skip
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    monkeypatch.setattr(claude_vision, "make_client", lambda: fake)
    return fake


# --- satellite STAC connectors: fixtures only, no network, no big downloads -------------------


class FakeStac:
    """Serves small committed STAC responses and writes tiny stand-in files for downloads."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, dict | None]] = []  # (method, url, body)
        self.downloads: list[tuple[str, dict]] = []  # (url, headers)
        self.responses = {
            "emit": ["stac_emit_p1.json", "stac_emit_p2.json"],
            "aster": ["stac_aster.json"],
            "enmap": ["stac_enmap_p1.json", "stac_enmap_p2.json"],
        }
        self.html_for: set[str] = set()  # downloads that return a login page instead of the file
        self.page: dict[str, int] = {}


@pytest.fixture
def stac(monkeypatch) -> FakeStac:
    import json

    from gossan.harvest.satellite.stac import StacSatelliteConnector

    fake = FakeStac()

    def request_json(self, method, url, body):
        fake.requests.append((method, url, body))
        # A fresh search starts at page 1; following a next link (GET, or POST with startIndex) advances.
        follows = method == "GET" or (body is not None and "startIndex" in body)
        fake.page[self.name] = fake.page.get(self.name, -1) + 1 if follows else 0
        pages = fake.responses[self.name]
        return json.loads((FIXTURES / pages[min(fake.page[self.name], len(pages) - 1)]).read_text(encoding="utf-8"))

    def download(self, url, part, headers):
        fake.downloads.append((url, headers))
        name = url.rsplit("/", 1)[-1].lower()
        if url in fake.html_for:
            part.write_bytes(b"<!DOCTYPE html><html>Please sign in</html>")
        elif name.endswith(".nc"):
            part.write_bytes(b"\x89HDF\r\n\x1a\n" + b"x" * 64)
        elif name.endswith((".tif", ".tiff")):
            part.write_bytes(b"II*\x00" + b"x" * 64)
        else:
            part.write_bytes(b"<?xml version='1.0'?><metadata/>")

    monkeypatch.setattr(StacSatelliteConnector, "_request_json", request_json)
    monkeypatch.setattr(StacSatelliteConnector, "_download", download)
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    monkeypatch.delenv("EOC_USERNAME", raising=False)
    monkeypatch.delenv("EOC_PASSWORD", raising=False)
    return fake


# --- US federal connectors: fixtures only, no network ----------------------------------------


class FakeUs:
    """Serves the committed MRDS GML and BLM ArcGIS pages; records every request."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []  # (service, url)
        self.mrds_file = "mrds_wfs.gml"
        self.blm_pages = ["blm_claims_p1.json", "blm_claims_p2.json", "blm_claims_p3.json", "blm_claims_p4.json", "blm_claims_empty.json"]
        self.blm_file_override: str | None = None  # serve this file for every ArcGIS request
        self.mrds_bytes: bytes | None = None  # serve these bytes instead of the file
        self.blm_features: list | None = None  # serve all of these in one page


@pytest.fixture
def us(monkeypatch) -> FakeUs:
    """Replace the MRDS and BLM network calls; no live requests in tests."""
    import json
    import re

    from gossan.harvest.us_federal.blm_mlrs import BlmClaimsConnector
    from gossan.harvest.us_federal.mrds import MrdsConnector

    fake = FakeUs()

    def mrds_get(self, url):
        fake.calls.append(("mrds", url))
        return fake.mrds_bytes if fake.mrds_bytes is not None else (FIXTURES / fake.mrds_file).read_bytes()

    def blm_get(self, url):
        fake.calls.append(("blm", url))
        if fake.blm_features is not None:
            return json.dumps({"type": "FeatureCollection", "features": fake.blm_features}).encode()
        if fake.blm_file_override:
            return (FIXTURES / fake.blm_file_override).read_bytes()
        offset = int(re.search(r"resultOffset=(\d+)", url).group(1))
        size = int(re.search(r"resultRecordCount=(\d+)", url).group(1))
        everything = [f for name in fake.blm_pages for f in json.loads((FIXTURES / name).read_text())["features"]]
        page = everything[offset : offset + size]  # what a real service does: the next slice of the same result
        body = {"type": "FeatureCollection", "features": page}
        if offset + size < len(everything):
            body["exceededTransferLimit"] = True
        return json.dumps(body).encode()

    monkeypatch.setattr(MrdsConnector, "_get", mrds_get)
    monkeypatch.setattr(BlmClaimsConnector, "_get", blm_get)
    return fake
