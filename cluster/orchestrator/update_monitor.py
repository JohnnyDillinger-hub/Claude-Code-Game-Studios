from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata as importlib_metadata
import json
import re
import socket
import subprocess
from pathlib import Path
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_VERSION_TOKEN_RE = re.compile(
    r"(?P<release>\d+(?:\.\d+)*)(?P<suffix>(?:[._-]?(?:dev|a|alpha|b|beta|rc|post)\d*)?)",
    re.IGNORECASE,
)
_VERSION_PREFIX_RE = re.compile(r"(?i)(?:^|[^0-9a-z])([vV]?\d+(?:\.\d+)*(?:[._-]?(?:dev|a|alpha|b|beta|rc|post)\d*)?)")
_VERSION_SUFFIX_ORDER = {
    "dev": 0,
    "a": 1,
    "alpha": 1,
    "b": 2,
    "beta": 2,
    "rc": 3,
    "": 4,
    "post": 5,
}


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _truncate_text(text: str | None, *, limit: int = 220) -> str | None:
    if text is None:
        return None
    normalized = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if not normalized:
        return None
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _release_summary(text: str | None, *, limit: int = 160) -> str | None:
    if text is None:
        return None
    normalized = " ".join(part.strip() for part in text.splitlines() if part.strip())
    if not normalized:
        return None
    first_sentence = normalized.split(". ", 1)[0]
    candidate = first_sentence if len(first_sentence) <= limit else normalized
    if len(candidate) <= limit:
        return candidate
    return candidate[: limit - 1].rstrip() + "…"


def normalize_version_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith(("git:", "sha256:", "python-package:", "command:", "workspace:")):
        return text
    matches = list(_VERSION_PREFIX_RE.finditer(text))
    if matches:
        candidate = matches[-1].group(1)
    else:
        candidate = text
    candidate = candidate.lstrip("vV")
    version_match = _VERSION_TOKEN_RE.search(candidate)
    if version_match is None:
        return candidate
    release = version_match.group("release")
    suffix = version_match.group("suffix") or ""
    return f"{release}{suffix}"


def _version_sort_key(value: str | None) -> tuple[tuple[int, ...], int, int] | None:
    normalized = normalize_version_text(value)
    if normalized is None:
        return None
    match = _VERSION_TOKEN_RE.fullmatch(normalized)
    if match is None:
        return None
    release = tuple(int(part) for part in match.group("release").split("."))
    suffix = (match.group("suffix") or "").lower()
    suffix_rank = _VERSION_SUFFIX_ORDER[""]
    suffix_number = 0
    if suffix:
        suffix_match = re.fullmatch(
            r"[._-]?(dev|a|alpha|b|beta|rc|post)(\d*)",
            suffix,
            re.IGNORECASE,
        )
        if suffix_match is not None:
            suffix_name = suffix_match.group(1).lower()
            suffix_rank = _VERSION_SUFFIX_ORDER.get(suffix_name, _VERSION_SUFFIX_ORDER[""])
            suffix_number = int(suffix_match.group(2) or 0)
        else:
            suffix_rank = _VERSION_SUFFIX_ORDER[""]
    return release, suffix_rank, suffix_number


def compare_versions(local_version: str | None, upstream_version: str | None) -> bool | None:
    local_key = _version_sort_key(local_version)
    upstream_key = _version_sort_key(upstream_version)
    if local_key is None or upstream_key is None:
        return None
    return local_key < upstream_key


@dataclass(frozen=True, slots=True)
class VersionSnapshot:
    source_kind: str
    source_name: str
    status: str
    raw_version: str | None = None
    normalized_version: str | None = None
    source_ref: str | None = None
    source_url: str | None = None
    summary: str | None = None
    notes: str | None = None
    observed_at: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source_kind": self.source_kind,
            "source_name": self.source_name,
            "status": self.status,
        }
        if self.raw_version is not None:
            payload["raw_version"] = self.raw_version
        if self.normalized_version is not None:
            payload["normalized_version"] = self.normalized_version
        if self.source_ref is not None:
            payload["source_ref"] = self.source_ref
        if self.source_url is not None:
            payload["source_url"] = self.source_url
        if self.summary is not None:
            payload["summary"] = self.summary
        if self.notes is not None:
            payload["notes"] = self.notes
        if self.observed_at is not None:
            payload["observed_at"] = self.observed_at
        if self.error is not None:
            payload["error"] = self.error
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VersionSnapshot":
        return cls(
            source_kind=str(payload["source_kind"]),
            source_name=str(payload["source_name"]),
            status=str(payload["status"]),
            raw_version=(
                str(payload["raw_version"]) if payload.get("raw_version") is not None else None
            ),
            normalized_version=(
                str(payload["normalized_version"])
                if payload.get("normalized_version") is not None
                else None
            ),
            source_ref=str(payload["source_ref"]) if payload.get("source_ref") else None,
            source_url=str(payload["source_url"]) if payload.get("source_url") else None,
            summary=str(payload["summary"]) if payload.get("summary") else None,
            notes=str(payload["notes"]) if payload.get("notes") else None,
            observed_at=str(payload["observed_at"]) if payload.get("observed_at") else None,
            error=str(payload["error"]) if payload.get("error") else None,
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class ComponentUpdateReport:
    component_id: str
    display_name: str
    category: str
    current: VersionSnapshot
    latest: VersionSnapshot | None
    update_available: bool | None
    status: str
    change_summary: str | None = None
    source_refs: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "component_id": self.component_id,
            "display_name": self.display_name,
            "category": self.category,
            "current": self.current.to_dict(),
            "status": self.status,
        }
        if self.latest is not None:
            payload["latest"] = self.latest.to_dict()
        if self.update_available is not None:
            payload["update_available"] = self.update_available
        if self.change_summary is not None:
            payload["change_summary"] = self.change_summary
        if self.source_refs:
            payload["source_refs"] = list(self.source_refs)
        if self.notes:
            payload["notes"] = list(self.notes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ComponentUpdateReport":
        return cls(
            component_id=str(payload["component_id"]),
            display_name=str(payload["display_name"]),
            category=str(payload["category"]),
            current=VersionSnapshot.from_dict(payload["current"]),
            latest=(
                VersionSnapshot.from_dict(payload["latest"])
                if payload.get("latest") is not None
                else None
            ),
            update_available=(
                bool(payload["update_available"])
                if payload.get("update_available") is not None
                else None
            ),
            status=str(payload["status"]),
            change_summary=(
                str(payload["change_summary"]) if payload.get("change_summary") else None
            ),
            source_refs=tuple(str(item) for item in payload.get("source_refs", [])),
            notes=tuple(str(item) for item in payload.get("notes", [])),
        )


@dataclass(frozen=True, slots=True)
class UpdateReport:
    generated_at: str
    allow_network: bool
    timeout_seconds: float
    repo_root: str
    components: tuple[ComponentUpdateReport, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> dict[str, int]:
        tracked = len(self.components)
        update_available = sum(1 for item in self.components if item.update_available is True)
        unavailable = sum(1 for item in self.components if item.status == "unavailable")
        errors = sum(1 for item in self.components if item.status == "error")
        unknown = sum(1 for item in self.components if item.status == "unknown")
        return {
            "tracked": tracked,
            "update_available": update_available,
            "unavailable": unavailable,
            "errors": errors,
            "unknown": unknown,
        }

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "generated_at": self.generated_at,
            "allow_network": self.allow_network,
            "timeout_seconds": self.timeout_seconds,
            "repo_root": self.repo_root,
            "summary": self.summary(),
            "components": [component.to_dict() for component in self.components],
        }
        if self.warnings:
            payload["warnings"] = list(self.warnings)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UpdateReport":
        return cls(
            generated_at=str(payload["generated_at"]),
            allow_network=bool(payload["allow_network"]),
            timeout_seconds=float(payload["timeout_seconds"]),
            repo_root=str(payload["repo_root"]),
            components=tuple(
                ComponentUpdateReport.from_dict(item)
                for item in payload.get("components", [])
            ),
            warnings=tuple(str(item) for item in payload.get("warnings", [])),
        )


class VersionSource(Protocol):
    source_kind: str
    source_name: str
    source_ref: str | None

    def resolve(
        self,
        *,
        allow_network: bool,
        timeout_seconds: float,
        repo_root: Path,
    ) -> VersionSnapshot: ...


@dataclass(frozen=True, slots=True)
class PythonPackageVersionSource:
    package_names: tuple[str, ...]
    source_name: str = "python-package"
    source_ref: str | None = None

    @classmethod
    def single(
        cls,
        package_name: str,
        *,
        source_ref: str | None = None,
    ) -> "PythonPackageVersionSource":
        return cls(package_names=(package_name,), source_ref=source_ref)

    def resolve(
        self,
        *,
        allow_network: bool,
        timeout_seconds: float,
        repo_root: Path,
    ) -> VersionSnapshot:
        del allow_network, timeout_seconds, repo_root
        errors: list[str] = []
        for package_name in self.package_names:
            try:
                raw_version = importlib_metadata.version(package_name)
                return VersionSnapshot(
                    source_kind="local",
                    source_name=self.source_name,
                    status="available",
                    raw_version=raw_version,
                    normalized_version=normalize_version_text(raw_version),
                    source_ref=self.source_ref or f"python-package:{package_name}",
                    metadata={"package_name": package_name},
                )
            except importlib_metadata.PackageNotFoundError:
                errors.append(f"{package_name}:not-installed")
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"{package_name}:{exc}")
        return VersionSnapshot(
            source_kind="local",
            source_name=self.source_name,
            status="unavailable",
            source_ref=self.source_ref or ",".join(self.package_names),
            error="; ".join(errors) if errors else "package not installed",
        )


@dataclass(frozen=True, slots=True)
class CommandVersionSource:
    command: tuple[str, ...]
    source_name: str
    source_ref: str | None = None

    def resolve(
        self,
        *,
        allow_network: bool,
        timeout_seconds: float,
        repo_root: Path,
    ) -> VersionSnapshot:
        del allow_network, repo_root
        try:
            completed = subprocess.run(
                list(self.command),
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:
            return VersionSnapshot(
                source_kind="local",
                source_name=self.source_name,
                status="unavailable",
                source_ref=self.source_ref or shlex_join(self.command),
                error=str(exc),
            )
        except subprocess.TimeoutExpired as exc:
            return VersionSnapshot(
                source_kind="local",
                source_name=self.source_name,
                status="unavailable",
                source_ref=self.source_ref or shlex_join(self.command),
                error=f"timeout after {timeout_seconds}s",
                metadata={"stdout": exc.stdout, "stderr": exc.stderr},
            )
        except subprocess.CalledProcessError as exc:
            text = (exc.stdout or "") + "\n" + (exc.stderr or "")
            raw_version = text.strip() or None
            return VersionSnapshot(
                source_kind="local",
                source_name=self.source_name,
                status="unavailable",
                raw_version=raw_version,
                normalized_version=normalize_version_text(raw_version),
                source_ref=self.source_ref or shlex_join(self.command),
                error=f"exit code {exc.returncode}",
            )
        text = (completed.stdout or completed.stderr or "").strip() or None
        return VersionSnapshot(
            source_kind="local",
            source_name=self.source_name,
            status="available",
            raw_version=text,
            normalized_version=normalize_version_text(text),
            source_ref=self.source_ref or shlex_join(self.command),
            metadata={
                "stdout": completed.stdout.strip() if completed.stdout else None,
                "stderr": completed.stderr.strip() if completed.stderr else None,
            },
        )


@dataclass(frozen=True, slots=True)
class WorkspaceFileVersionSource:
    relative_path: str
    source_name: str = "workspace-file"
    source_ref: str | None = None

    def resolve(
        self,
        *,
        allow_network: bool,
        timeout_seconds: float,
        repo_root: Path,
    ) -> VersionSnapshot:
        del allow_network, timeout_seconds
        path = (repo_root / self.relative_path).resolve()
        if not path.exists():
            return VersionSnapshot(
                source_kind="local",
                source_name=self.source_name,
                status="unavailable",
                source_ref=self.source_ref or str(path),
                error="file not found",
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        git_head = _git_head(repo_root)
        metadata: dict[str, Any] = {
            "path": str(path),
            "sha256": digest,
        }
        if git_head is not None:
            metadata["git_head"] = git_head
        raw_version = git_head or digest
        normalized_version = f"git:{git_head}" if git_head is not None else f"sha256:{digest}"
        return VersionSnapshot(
            source_kind="local",
            source_name=self.source_name,
            status="available",
            raw_version=raw_version,
            normalized_version=normalized_version,
            source_ref=self.source_ref or str(path),
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class GitHubReleaseVersionSource:
    repository: str
    source_name: str = "github-release"
    source_ref: str | None = None

    def resolve(
        self,
        *,
        allow_network: bool,
        timeout_seconds: float,
        repo_root: Path,
    ) -> VersionSnapshot:
        del repo_root
        if not allow_network:
            return VersionSnapshot(
                source_kind="upstream",
                source_name=self.source_name,
                status="unavailable",
                source_ref=self.source_ref or f"github:{self.repository}",
                error="network disabled",
            )
        url = f"https://api.github.com/repos/{self.repository}/releases/latest"
        try:
            payload = _fetch_json(url, timeout_seconds=timeout_seconds)
        except (HTTPError, URLError, TimeoutError, socket.timeout, ValueError) as exc:
            return VersionSnapshot(
                source_kind="upstream",
                source_name=self.source_name,
                status="unavailable",
                source_ref=self.source_ref or f"github:{self.repository}",
                source_url=url,
                error=str(exc),
            )
        raw_version = str(payload.get("tag_name") or payload.get("name") or "")
        summary = _release_summary(str(payload.get("body") or payload.get("name") or raw_version))
        normalized_version = normalize_version_text(raw_version)
        notes = _truncate_text(str(payload.get("body") or "")) if payload.get("body") else None
        return VersionSnapshot(
            source_kind="upstream",
            source_name=self.source_name,
            status="available",
            raw_version=raw_version or None,
            normalized_version=normalized_version,
            source_ref=self.source_ref or f"github:{self.repository}",
            source_url=str(payload.get("html_url") or url),
            summary=summary,
            notes=notes,
            observed_at=(
                str(payload.get("published_at")) if payload.get("published_at") else None
            ),
            metadata={
                "repository": self.repository,
                "draft": bool(payload.get("draft")),
                "prerelease": bool(payload.get("prerelease")),
            },
        )


@dataclass(frozen=True, slots=True)
class ComponentTracker:
    component_id: str
    display_name: str
    category: str
    local_source: VersionSource
    upstream_source: VersionSource | None = None
    source_refs: tuple[str, ...] = field(default_factory=tuple)

    def collect(
        self,
        *,
        allow_network: bool,
        timeout_seconds: float,
        repo_root: Path,
    ) -> ComponentUpdateReport:
        notes: list[str] = []
        current = self.local_source.resolve(
            allow_network=allow_network,
            timeout_seconds=timeout_seconds,
            repo_root=repo_root,
        )
        latest = (
            self.upstream_source.resolve(
                allow_network=allow_network,
                timeout_seconds=timeout_seconds,
                repo_root=repo_root,
            )
            if self.upstream_source is not None
            else None
        )
        update_available: bool | None = None
        status = "unknown"
        if current.status == "available" and latest is not None:
            if latest.status == "available":
                update_available = compare_versions(
                    current.normalized_version,
                    latest.normalized_version,
                )
                if update_available is True:
                    status = "update-available"
                elif update_available is False:
                    status = "up-to-date"
                else:
                    status = "unknown"
            else:
                status = "unavailable"
                notes.append(latest.error or "upstream unavailable")
        elif current.status != "available":
            status = "unavailable"
            notes.append(current.error or "local version unavailable")
        elif latest is None:
            status = "up-to-date"

        change_summary = None
        if latest is not None and latest.status == "available":
            change_summary = latest.summary or latest.notes

        if current.status == "error" or (latest is not None and latest.status == "error"):
            status = "error"

        return ComponentUpdateReport(
            component_id=self.component_id,
            display_name=self.display_name,
            category=self.category,
            current=current,
            latest=latest,
            update_available=update_available,
            status=status,
            change_summary=change_summary,
            source_refs=self._resolved_source_refs(current=current, latest=latest),
            notes=tuple(notes),
        )

    def _resolved_source_refs(
        self,
        *,
        current: VersionSnapshot,
        latest: VersionSnapshot | None,
    ) -> tuple[str, ...]:
        refs: list[str] = []
        if current.source_ref:
            refs.append(current.source_ref)
        if latest is not None and latest.source_ref:
            refs.append(latest.source_ref)
        if self.source_refs:
            refs.extend(self.source_refs)
        seen: set[str] = set()
        deduped: list[str] = []
        for ref in refs:
            if ref in seen:
                continue
            seen.add(ref)
            deduped.append(ref)
        return tuple(deduped)


@dataclass(frozen=True, slots=True)
class UpdateMonitor:
    trackers: tuple[ComponentTracker, ...]
    repo_root: Path

    def generate_report(
        self,
        *,
        allow_network: bool = True,
        timeout_seconds: float = 5.0,
    ) -> UpdateReport:
        components: list[ComponentUpdateReport] = []
        warnings: list[str] = []
        for tracker in self.trackers:
            try:
                components.append(
                    tracker.collect(
                        allow_network=allow_network,
                        timeout_seconds=timeout_seconds,
                        repo_root=self.repo_root,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                warnings.append(f"{tracker.component_id}: {exc}")
                components.append(
                    ComponentUpdateReport(
                        component_id=tracker.component_id,
                        display_name=tracker.display_name,
                        category=tracker.category,
                        current=VersionSnapshot(
                            source_kind="local",
                            source_name="unknown",
                            status="error",
                            error=str(exc),
                        ),
                        latest=None,
                        update_available=None,
                        status="error",
                        notes=(str(exc),),
                    )
                )
        return UpdateReport(
            generated_at=utc_now().isoformat().replace("+00:00", "Z"),
            allow_network=allow_network,
            timeout_seconds=timeout_seconds,
            repo_root=str(self.repo_root),
            components=tuple(components),
            warnings=tuple(warnings),
        )


def build_default_update_monitor(repo_root: str | Path | None = None) -> UpdateMonitor:
    resolved_repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    trackers = (
        ComponentTracker(
            component_id="vllm",
            display_name="vLLM",
            category="runtime",
            local_source=PythonPackageVersionSource(
                package_names=("vllm",),
                source_ref="python-package:vllm",
            ),
            upstream_source=GitHubReleaseVersionSource(
                repository="vllm-project/vllm",
                source_ref="https://github.com/vllm-project/vllm/releases/latest",
            ),
            source_refs=("https://github.com/vllm-project/vllm",),
        ),
        ComponentTracker(
            component_id="sglang",
            display_name="SGLang",
            category="runtime",
            local_source=PythonPackageVersionSource(
                package_names=("sglang",),
                source_ref="python-package:sglang",
            ),
            upstream_source=GitHubReleaseVersionSource(
                repository="sgl-project/sglang",
                source_ref="https://github.com/sgl-project/sglang/releases/latest",
            ),
            source_refs=("https://github.com/sgl-project/sglang",),
        ),
        ComponentTracker(
            component_id="deepspeed",
            display_name="DeepSpeed",
            category="runtime",
            local_source=PythonPackageVersionSource(
                package_names=("deepspeed",),
                source_ref="python-package:deepspeed",
            ),
            upstream_source=GitHubReleaseVersionSource(
                repository="deepspeedai/DeepSpeed",
                source_ref="https://github.com/deepspeedai/DeepSpeed/releases/latest",
            ),
            source_refs=("https://github.com/deepspeedai/DeepSpeed",),
        ),
        ComponentTracker(
            component_id="tensorrt-llm",
            display_name="TensorRT-LLM",
            category="runtime",
            local_source=PythonPackageVersionSource(
                package_names=("tensorrt-llm", "tensorrt_llm"),
                source_ref="python-package:tensorrt-llm",
            ),
            upstream_source=GitHubReleaseVersionSource(
                repository="NVIDIA/TensorRT-LLM",
                source_ref="https://github.com/NVIDIA/TensorRT-LLM/releases/latest",
            ),
            source_refs=("https://github.com/NVIDIA/TensorRT-LLM",),
        ),
        ComponentTracker(
            component_id="ollama",
            display_name="Ollama",
            category="runtime",
            local_source=CommandVersionSource(
                command=("ollama", "--version"),
                source_name="ollama-cli",
                source_ref="command:ollama --version",
            ),
            upstream_source=GitHubReleaseVersionSource(
                repository="ollama/ollama",
                source_ref="https://github.com/ollama/ollama/releases/latest",
            ),
            source_refs=("https://github.com/ollama/ollama", "https://ollama.com"),
        ),
        ComponentTracker(
            component_id="runtime-profile-catalog",
            display_name="Runtime Profile Catalog",
            category="workspace-config",
            local_source=WorkspaceFileVersionSource(
                relative_path="cluster/orchestrator/model_profiles.yaml",
                source_ref="cluster/orchestrator/model_profiles.yaml",
            ),
            upstream_source=None,
            source_refs=(str((resolved_repo_root / "cluster/orchestrator/model_profiles.yaml").resolve()),),
        ),
    )
    return UpdateMonitor(trackers=trackers, repo_root=resolved_repo_root)


def render_update_report_text(report: UpdateReport) -> str:
    lines: list[str] = []
    summary = report.summary()
    lines.append(
        f"Update report generated_at={report.generated_at} network={'on' if report.allow_network else 'off'} "
        f"timeout={report.timeout_seconds:.1f}s tracked={summary['tracked']} "
        f"updates={summary['update_available']} unavailable={summary['unavailable']} "
        f"errors={summary['errors']} unknown={summary['unknown']}"
    )
    if report.warnings:
        lines.append("Warnings:")
        for warning in report.warnings:
            lines.append(f"  - {warning}")
    for component in report.components:
        current_version = component.current.normalized_version or component.current.raw_version or "unknown"
        latest_version = "-"
        if component.latest is not None:
            latest_version = component.latest.normalized_version or component.latest.raw_version or "unknown"
        availability = "yes" if component.update_available else ("no" if component.update_available is False else "unknown")
        line = (
            f"- {component.display_name} [{component.category}] "
            f"local={current_version} latest={latest_version} update={availability} status={component.status}"
        )
        lines.append(line)
        if component.change_summary:
            lines.append(f"  summary: {component.change_summary}")
        if component.notes:
            for note in component.notes:
                lines.append(f"  note: {note}")
        if component.source_refs:
            lines.append(f"  refs: {', '.join(component.source_refs)}")
    return "\n".join(lines)


def _fetch_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "Claude-Code-Game-Studios-update-monitor/1.0",
        },
        method="GET",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read()
    parsed = json.loads(payload.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object")
    return parsed


def _git_head(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def shlex_join(command: Sequence[str]) -> str:
    return " ".join(_quote_shlex(part) for part in command)


def _quote_shlex(value: str) -> str:
    if not value:
        return "''"
    if re.fullmatch(r"[A-Za-z0-9_@%+=:,./-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"
