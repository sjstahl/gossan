"""Append-only provenance log (``provenance.jsonl``), one JSON object per line."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from gossan import __version__

PROVENANCE_FILE = "provenance.jsonl"


class ProvenanceRecord(BaseModel):
    """One dataset or operation, following the schema in PLAN.md section 6."""

    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(min_length=1)
    source: str = Field(min_length=1)
    identifier: str
    url: str | None
    retrieved_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    license: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    gossan_version: str = __version__


class ProvenanceLog:
    """Reads and appends records in a project's ``provenance.jsonl``."""

    def __init__(self, project_dir: Path) -> None:
        self.path = Path(project_dir) / PROVENANCE_FILE

    def append(self, record: ProvenanceRecord) -> None:
        """Validate (already done by the model) and append one line."""
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(record.model_dump_json() + "\n")

    def log(self, **fields: Any) -> ProvenanceRecord:
        """Build a record from keyword fields, append it and return it."""
        record = ProvenanceRecord(**fields)
        self.append(record)
        return record

    def read(self) -> list[ProvenanceRecord]:
        """Return all records; raises ``ValueError`` on a malformed line."""
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [ProvenanceRecord.model_validate_json(line) for line in lines if line.strip()]
