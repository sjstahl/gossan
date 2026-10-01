"""Review queue: things a person must check before they enter the project data.

A plain CSV, ``digitize/review_queue.csv``, so it can be opened in a spreadsheet and
diffed. Nothing is added to it automatically yet; later pipeline steps (OCR text, tables,
georeferenced maps) will call ``add_item`` and a reviewer will move items out of PENDING.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from gossan.core.project import Project

QUEUE_FILE = "review_queue.csv"


class ItemType(StrEnum):
    TEXT = "text"
    MAP = "map"
    TABLE = "table"


class Status(StrEnum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass
class ReviewItem:
    item_id: str
    created_at: str
    report: str  # ARIS number
    type: str
    page: str  # page number or range, blank if the item is the whole report
    status: str
    source: str  # file to review (e.g. the extracted CSV), relative to the project folder
    note: str
    image: str = ""  # page image to review it against, relative to the project folder


COLUMNS = [f.name for f in fields(ReviewItem)]


def queue_path(project: Project) -> Path:
    return project.root / "digitize" / QUEUE_FILE


def init_queue(project: Project) -> Path:
    """Create the queue file with its header if it does not exist yet.

    Never discards items; a queue written with an older set of columns is upgraded in place.
    """
    path = queue_path(project)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_all(path, [])
    else:
        with path.open(encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), [])
        if header != COLUMNS:
            _write_all(path, read_queue(project))
    return path


def _write_all(path: Path, items: list[ReviewItem]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(i) for i in items)


def read_queue(project: Project) -> list[ReviewItem]:
    """All items, oldest first; an empty list if the queue was never initialised."""
    path = queue_path(project)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return [ReviewItem(**{c: row.get(c) or "" for c in COLUMNS}) for row in csv.DictReader(f)]


def add_item(
    project: Project,
    report: int | str,
    item_type: ItemType | str,
    *,
    page: int | str = "",
    source: str = "",
    note: str = "",
    image: str = "",
) -> ReviewItem | None:
    """Append a PENDING item; returns None if an identical PENDING item is already queued."""
    item_type = ItemType(item_type)  # rejects unknown types
    path = init_queue(project)
    items = read_queue(project)
    key = (str(report), item_type.value, str(page), source)
    if any(
        (i.report, i.type, i.page, i.source) == key and i.status == Status.PENDING for i in items
    ):
        return None
    item = ReviewItem(
        item_id=f"R{len(items) + 1:04d}",
        created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        report=str(report),
        type=item_type.value,
        page=str(page),
        status=Status.PENDING.value,
        source=source,
        note=note,
        image=image,
    )
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=COLUMNS).writerow(asdict(item))
    project.provenance.log(
        dataset="review_queue",
        source="gossan",
        identifier=item.item_id,
        url=None,
        license="N/A (workflow record)",
        operation="queue_item",
        parameters={k: v for k, v in asdict(item).items() if k != "item_id"},
    )
    return item


def summarize(items: list[ReviewItem]) -> dict[str, dict[str, int]]:
    """Counts by status then type, e.g. ``{"PENDING": {"map": 2, "text": 1}}``."""
    out: dict[str, dict[str, int]] = {}
    for i in items:
        by_type = out.setdefault(i.status, {})
        by_type[i.type] = by_type.get(i.type, 0) + 1
    return out
