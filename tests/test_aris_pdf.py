import hashlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gossan.cli import app
from gossan.core.project import Project
from gossan.harvest.bc.aris_pdf import pdf_url

runner = CliRunner()
URL_8038 = "https://apps.nrs.gov.bc.ca/pub/aris/Report/08038.pdf/"
URL_724 = "https://apps.nrs.gov.bc.ca/pub/aris/Report/00724.pdf/"


@pytest.fixture
def project(aoi_file, wfs) -> Project:
    """A project with the (mocked) ARIS index harvested: reports 8038 (1979) and 724 (1964)."""
    assert runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)]).exit_code == 0
    assert runner.invoke(app, ["harvest", "demo", "--source", "aris_index"]).exit_code == 0
    return Project(Path("demo"))


def pdf_files(project: Project) -> list[str]:
    folder = project.raw_dir / "aris_pdfs"
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def test_pdf_url_from_report_url():
    assert pdf_url("https://apps.nrs.gov.bc.ca/pub/aris/detail/8038") == URL_8038
    assert pdf_url("https://apps.nrs.gov.bc.ca/pub/aris/detail/42141") == (
        "https://apps.nrs.gov.bc.ca/pub/aris/Report/42141.pdf/"
    )
    with pytest.raises(ValueError):
        pdf_url("https://example.test/not-aris")


def test_default_downloads_no_pdfs(project, aris_pdfs):
    result = runner.invoke(app, ["harvest", "demo", "--source", "aris_index"])
    assert result.exit_code == 0, result.output
    assert aris_pdfs.calls == []
    assert pdf_files(project) == []
    assert "aris_pdfs" not in result.output


def test_limit_downloads_newest_first_and_logs_each(project, aris_pdfs, fixtures_dir):
    result = runner.invoke(
        app, ["harvest", "demo", "--source", "aris_index", "--limit-pdfs", "1"]
    )
    assert result.exit_code == 0, result.output
    assert "1 downloaded" in result.output
    assert "1 more not fetched" in result.output

    assert aris_pdfs.calls == [URL_8038]  # 1979 report before the 1964 one
    assert pdf_files(project) == ["ARIS_08038.pdf"]  # no leftover .part files

    (record,) = [r for r in project.provenance.read() if r.operation == "fetch_pdf"]
    assert record.dataset == "aris_pdfs"
    assert record.identifier == "ARIS 8038"
    assert record.url == URL_8038
    assert "Access Only" in record.license
    assert record.parameters["file"] == "raw/aris_pdfs/ARIS_08038.pdf"
    dummy = (fixtures_dir / "dummy_report.pdf").read_bytes()
    assert record.parameters["sha256"] == hashlib.sha256(dummy).hexdigest()
    assert record.parameters["bytes"] == len(dummy)


def test_existing_pdfs_are_skipped_not_replaced(project, aris_pdfs):
    args = ["harvest", "demo", "--source", "aris_index", "--limit-pdfs"]
    runner.invoke(app, [*args, "1"])
    original = project.raw_dir / "aris_pdfs" / "ARIS_08038.pdf"
    original.write_bytes(b"%PDF-1.4 edited by hand")  # must survive a re-run

    result = runner.invoke(app, [*args, "5"])
    assert result.exit_code == 0, result.output
    assert "1 downloaded, 1 already present" in result.output
    assert aris_pdfs.calls == [URL_8038, URL_724]
    assert original.read_bytes() == b"%PDF-1.4 edited by hand"
    assert pdf_files(project) == ["ARIS_00724.pdf", "ARIS_08038.pdf"]


def test_failures_are_reported_and_leave_no_files(project, aris_pdfs):
    aris_pdfs.missing.add(URL_8038)  # no scanned PDF
    aris_pdfs.not_pdf.add(URL_724)  # a login or error page instead of a PDF
    result = runner.invoke(
        app, ["harvest", "demo", "--source", "aris_index", "--limit-pdfs", "5"]
    )
    assert result.exit_code == 0, result.output
    assert "0 downloaded" in result.output
    assert "2 failed" in result.output
    assert "ARIS 8038" in result.output and "ARIS 724" in result.output
    assert pdf_files(project) == []
    assert not [r for r in project.provenance.read() if r.operation == "fetch_pdf"]


def test_limit_without_index_is_an_error(aoi_file, wfs, aris_pdfs):
    runner.invoke(app, ["new", "demo", "--aoi", str(aoi_file)])
    result = runner.invoke(app, ["harvest", "demo", "--source", "minfile", "--limit-pdfs", "1"])
    assert result.exit_code == 1
    assert "aris_index" in result.output
    assert aris_pdfs.calls == []
