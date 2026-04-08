from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile

from cluster.providers.models import ProvisionJob


class ProvisionJobStore:
    """Simple JSON-backed state store for provider provisioning jobs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> list[ProvisionJob]:
        if not self.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return [ProvisionJob.from_dict(item) for item in payload.get("jobs", [])]

    def save(self, jobs: list[ProvisionJob]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "jobs": [job.to_dict() for job in jobs]}
        data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(self.path.parent),
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, self.path)

    def upsert(self, job: ProvisionJob) -> None:
        jobs = self.load()
        filtered = [existing for existing in jobs if existing.job_id != job.job_id]
        filtered.append(job)
        filtered.sort(key=lambda item: item.created_at)
        self.save(filtered)
