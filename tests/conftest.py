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
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(claude_vision, "make_client", lambda: fake)
    return fake
