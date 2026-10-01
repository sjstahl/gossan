"""Drillhole and assay tables from scanned pages, read by Claude vision.

This sends page images to OpenRouter (which forwards them to Anthropic), which costs money
and sends page content to two third parties. It only runs when asked for explicitly
(``--extract-scanned-tables``) on pages you name (``--target-pages``). Credentials come from
the ``OPENROUTER_API_KEY`` environment variable; the model from ``GOSSAN_CLAUDE_MODEL``
(default ``DEFAULT_MODEL``, in OpenRouter's ``vendor/model`` format).

OpenRouter's ``https://openrouter.ai/api/v1`` is its OpenAI-style chat endpoint. The Anthropic
SDK talks the Messages protocol, which OpenRouter serves at ``https://openrouter.ai/api`` (the
SDK adds ``/v1/messages`` itself), so that is the base URL used here, with the key sent as a
bearer token. The structured-output call (``messages.parse``) has not been run against
OpenRouter from this code; OpenRouter lists structured-output support for the Claude models.

The reply is constrained to the ``PageTable`` schema (structured outputs), so it always
parses, but the *values* are a model's reading of a scan and can be wrong (digits, decimal
points, merged cells, detection-limit flags). Every extracted table is added to the review
queue as PENDING with the page image and the CSV, and must be checked by a person before it
goes into a drillhole database.

The API schema carries ``attributes`` as a list of name/value pairs because structured
outputs do not accept free-form objects; ``table_to_rows`` turns them into the dict the
rest of Gossan uses. Depths are read as printed and converted to metres here, in code,
when the page says feet.
"""

from __future__ import annotations

import base64
import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, Field

from gossan.core.project import Project
from gossan.digitize import review
from gossan.digitize.pages import DEFAULT_DPI, page_image_path, render_page
from gossan.digitize.text import DERIVED_LICENSE

MODEL_ENV = "GOSSAN_CLAUDE_MODEL"
KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
DEFAULT_MODEL = "anthropic/claude-opus-5.5"
PROVIDER = "OpenRouter (Claude vision)"
#: Hard ceiling on pages sent in one run, so a typo in --target-pages cannot run up a bill.
MAX_PAGES_PER_RUN = 25
FEET_TO_M = 0.3048

SYSTEM_PROMPT = """You transcribe drillhole and assay tables from scanned mineral exploration \
assessment reports. Copy what is printed; never infer, correct, round or complete values.

- One row per printed table row (a sample interval, a lithology interval, or a collar/survey line).
- hole_id: the hole, drill-hole or sample-series identifier. If it is blank because it repeats \
the row above, fill it in.
- from_m / to_m: the interval depths as printed, as plain numbers, in the unit the table uses \
(do not convert; state the unit in depth_unit). Use null when the row has no interval \
(for example a collar row) or the cell is blank or illegible.
- attributes: every other column as a name/value pair. Use the column heading, with its unit \
if printed, as the name (for example "Au (ppb)", "Cu (ppm)", "Lithology", "Azimuth"). Give \
values exactly as printed, as text, including detection-limit flags such as "<5" and "nil". \
Use an empty string for a blank cell and "?" for an illegible one.
- table_type: assay, lithology, collar, survey, or other.
- If the page has no drillhole, sample or assay table, return no rows, table_type "other", \
and say what the page is in notes.
- notes: anything a reviewer should know (smudged areas, rotated text, a table split across \
pages, columns you could not read)."""


class ClaudeVisionError(Exception):
    """The API could not be used for this page; the message says why."""


class Attribute(BaseModel):
    name: str = Field(description="Column heading, with its unit if printed")
    value: str = Field(description="Cell value exactly as printed")


class TableRow(BaseModel):
    hole_id: str
    from_m: float | None = Field(description="Interval start as printed (see depth_unit)")
    to_m: float | None = Field(description="Interval end as printed (see depth_unit)")
    attributes: list[Attribute]


class PageTable(BaseModel):
    table_type: Literal["assay", "lithology", "collar", "survey", "other"]
    depth_unit: Literal["m", "ft", "unknown"] = Field(
        description="Unit of from_m / to_m as printed on the page"
    )
    rows: list[TableRow]
    notes: str


def make_client() -> anthropic.Anthropic:
    """An Anthropic SDK client routed through OpenRouter, using ``OPENROUTER_API_KEY``.

    The key goes in the ``Authorization: Bearer`` header; ``api_key=None`` keeps the SDK from
    also sending a stray ``ANTHROPIC_API_KEY`` from the environment to OpenRouter.
    """
    key = os.environ.get(KEY_ENV)
    if not key:
        raise ClaudeVisionError(f"{KEY_ENV} is not set")
    return anthropic.Anthropic(base_url=OPENROUTER_BASE_URL, auth_token=key, api_key=None)


def model_name() -> str:
    return os.environ.get(MODEL_ENV) or DEFAULT_MODEL


def read_page_image(client: anthropic.Anthropic, image: Path, model: str) -> tuple[PageTable, Any]:
    """Send one page image; returns the parsed table and the API usage."""
    data = base64.standard_b64encode(image.read_bytes()).decode("ascii")
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": data},
                        },
                        {"type": "text", "text": "Transcribe the table on this page."},
                    ],
                }
            ],
            output_format=PageTable,
        )
    except anthropic.APIError as e:
        raise ClaudeVisionError(f"{type(e).__name__}: {e}") from e
    if response.stop_reason in {"refusal", "max_tokens"}:
        raise ClaudeVisionError(f"the model stopped early ({response.stop_reason})")
    if response.parsed_output is None:
        raise ClaudeVisionError("the reply did not match the table schema")
    return response.parsed_output, response.usage


@dataclass
class ExtractedRows:
    columns: list[str]
    rows: list[dict[str, str]]
    converted_from_feet: bool


def table_to_rows(table: PageTable) -> ExtractedRows:
    """Flatten a ``PageTable`` to CSV rows: hole_id, from_m, to_m, then attribute columns.

    Attribute columns appear in order of first use; a repeated name within one row becomes
    ``name_2``. Depths printed in feet are converted to metres (3 decimals).
    """
    feet = table.depth_unit == "ft"
    columns = ["hole_id", "from_m", "to_m"]
    rows = []
    for r in table.rows:
        row = {"hole_id": r.hole_id}
        for key, depth in (("from_m", r.from_m), ("to_m", r.to_m)):
            if depth is None:
                row[key] = ""
            else:
                row[key] = f"{depth * FEET_TO_M:.3f}" if feet else f"{depth:g}"
        seen: dict[str, int] = {}
        for a in r.attributes:
            seen[a.name] = seen.get(a.name, 0) + 1
            name = a.name if seen[a.name] == 1 else f"{a.name}_{seen[a.name]}"
            if name not in columns:
                columns.append(name)
            row[name] = a.value
        rows.append(row)
    return ExtractedRows(columns, rows, feet)


def scanned_dir(project: Project) -> Path:
    return project.root / "digitize" / "tables" / "scanned"


@dataclass
class ScannedResult:
    written: list[Path] = field(default_factory=list)
    no_table: list[int] = field(default_factory=list)
    already_done: list[int] = field(default_factory=list)
    failed: dict[int, str] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0


def extract_scanned_tables(
    project: Project,
    aris_number: int,
    pdf: Path,
    pages: list[int],
    client: anthropic.Anthropic | None = None,
) -> ScannedResult:
    """Read the table on each of ``pages`` with Claude and queue the results for review.

    A page that already has a scanned-table CSV is skipped (delete the file to redo it) so a
    re-run does not pay twice. Raises ``ClaudeVisionError`` before spending anything if the
    API cannot be used or too many pages are requested; per-page failures are collected.
    """
    if len(pages) > MAX_PAGES_PER_RUN:
        raise ClaudeVisionError(
            f"{len(pages)} pages requested; the limit is {MAX_PAGES_PER_RUN} per run "
            "(narrow --target-pages)"
        )
    result = ScannedResult()
    out_dir = scanned_dir(project)
    todo = []
    for page in pages:
        if (out_dir / f"{aris_number:05d}_p{page}.csv").exists():
            result.already_done.append(page)
        else:
            todo.append(page)
    if not todo:
        return result

    client = client or make_client()
    model = model_name()
    for page in todo:
        csv_path = out_dir / f"{aris_number:05d}_p{page}.csv"
        image = page_image_path(project, aris_number, page)
        try:
            render_page(pdf, page, image, DEFAULT_DPI)
            table, usage = read_page_image(client, image, model)
        except (ClaudeVisionError, OSError, ValueError, RuntimeError) as e:
            result.failed[page] = str(e)
            continue
        result.input_tokens += usage.input_tokens
        result.output_tokens += usage.output_tokens
        flat = table_to_rows(table)
        params = {
            "input": f"raw/aris_pdfs/{pdf.name}",
            "page": page,
            "page_image": image.relative_to(project.root).as_posix(),
            "model": model,
            "provider": "openrouter",
            "table_type": table.table_type,
            "depth_unit_printed": table.depth_unit,
            "converted_feet_to_m": flat.converted_from_feet,
            "rows": len(flat.rows),
            "notes": table.notes,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "reviewed": False,
        }
        if not flat.rows:
            result.no_table.append(page)
            project.provenance.log(
                dataset="aris_tables_scanned",
                source=PROVIDER,
                identifier=f"ARIS {aris_number} p{page}",
                url=None,
                license=DERIVED_LICENSE,
                operation="extract_scanned_table_none",
                parameters=params,
            )
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat.columns, restval="")
            writer.writeheader()
            writer.writerows(flat.rows)
        rel_csv = csv_path.relative_to(project.root).as_posix()
        project.provenance.log(
            dataset="aris_tables_scanned",
            source=PROVIDER,
            identifier=f"ARIS {aris_number} p{page}",
            url=None,
            license=DERIVED_LICENSE,
            operation="extract_scanned_table",
            parameters={**params, "output": rel_csv},
        )
        review.add_item(
            project,
            aris_number,
            review.ItemType.TABLE,
            page=page,
            source=rel_csv,
            image=image.relative_to(project.root).as_posix(),
            note=(
                f"{table.table_type} table, {len(flat.rows)} row(s), read by {model}; "
                f"check against the page image. {table.notes}"
            ).strip(),
        )
        result.written.append(csv_path)
    return result
