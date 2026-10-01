"""Review queue: things a person must check before they enter the project data.

A plain CSV, ``digitize/review_queue.csv``, so it can be opened in a spreadsheet and
diffed. Pipeline steps add items as PENDING (Claude-read tables, georeferenced maps); a person
moves each to APPROVED or REJECTED with ``set_status`` (``gossan review approve|reject``).
Every decision is logged to provenance with who made it. A decided item is final: it cannot
be flipped, because an approved table may already be in the drillhole tables.
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
    reviewed_at: str = ""
    reviewed_by: str = ""
    decision_note: str = ""


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


class ReviewError(Exception):
    """A review action cannot be done; the message says why."""


def get_item(project: Project, item_id: str) -> ReviewItem:
    for item in read_queue(project):
        if item.item_id == item_id:
            return item
    raise ReviewError(f"no review item {item_id!r}; see 'gossan review list'")


def approved_items(project: Project, reports: set[str] | None = None) -> list[ReviewItem]:
    """APPROVED items, optionally only for the given ARIS numbers (as strings)."""
    return [
        i
        for i in read_queue(project)
        if i.status == Status.APPROVED and (reports is None or i.report in reports)
    ]


def set_status(
    project: Project,
    item_id: str,
    status: Status,
    *,
    by: str,
    note: str = "",
    details: dict[str, object] | None = None,
) -> ReviewItem:
    """Move a PENDING item to APPROVED or REJECTED, recording who decided and when.

    ``details`` (for example what was loaded into the database) goes into the provenance
    record. Raises ``ReviewError`` for an unknown or already decided item.
    """
    if status not in (Status.APPROVED, Status.REJECTED):
        raise ReviewError(f"cannot set status {status}")
    items = read_queue(project)
    target = next((i for i in items if i.item_id == item_id), None)
    if target is None:
        raise ReviewError(f"no review item {item_id!r}; see 'gossan review list'")
    if target.status != Status.PENDING:
        raise ReviewError(f"{item_id} is already {target.status}; decisions are final")
    previous = target.status
    target.status = status.value
    target.reviewed_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    target.reviewed_by = by
    target.decision_note = note
    _write_all(queue_path(project), items)
    project.provenance.log(
        dataset="review_queue",
        source="manual review",
        identifier=item_id,
        url=None,
        retrieved_at=datetime.now(UTC),
        license="N/A (workflow record)",
        operation="review_approve" if status == Status.APPROVED else "review_reject",
        parameters={
            "type": target.type,
            "report": target.report,
            "page": target.page,
            "source": target.source,
            "previous_status": previous,
            "reviewed_by": by,
            "decision_note": note,
            **(details or {}),
        },
    )
    return target
