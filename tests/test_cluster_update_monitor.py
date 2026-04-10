from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from cluster.orchestrator.clusterctl import main as clusterctl_main
from cluster.orchestrator.update_monitor import (
    ComponentTracker,
    ComponentUpdateReport,
    GitHubReleaseVersionSource,
    UpdateMonitor,
    UpdateReport,
    VersionSnapshot,
    build_default_update_monitor,
    compare_versions,
    normalize_version_text,
    render_update_report_text,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class StaticVersionSource:
    source_kind: str
    source_name: str
    source_ref: str | None
    snapshot: VersionSnapshot

    def resolve(
        self,
        *,
        allow_network: bool,
        timeout_seconds: float,
        repo_root: Path,
    ) -> VersionSnapshot:
        del allow_network, timeout_seconds, repo_root
        return self.snapshot


class ClusterUpdateMonitorTests(unittest.TestCase):
    def test_version_normalization_and_comparison(self) -> None:
        self.assertEqual(normalize_version_text("v0.19.0"), "0.19.0")
        self.assertEqual(normalize_version_text("TensorRT-LLM 1.2.0"), "1.2.0")
        self.assertTrue(compare_versions("0.18.9", "0.19.0"))
        self.assertFalse(compare_versions("0.19.0", "0.18.9"))
        self.assertIsNone(compare_versions("git:abc123", "0.19.0"))

    def test_default_monitor_tracks_expected_components(self) -> None:
        monitor = build_default_update_monitor(REPO_ROOT)
        component_ids = [tracker.component_id for tracker in monitor.trackers]

        self.assertEqual(
            component_ids,
            [
                "vllm",
                "sglang",
                "deepspeed",
                "tensorrt-llm",
                "ollama",
                "runtime-profile-catalog",
            ],
        )

    def test_offline_lookup_is_best_effort_and_serialization_is_stable(self) -> None:
        local_snapshot = VersionSnapshot(
            source_kind="local",
            source_name="static-local",
            status="available",
            raw_version="0.19.0",
            normalized_version="0.19.0",
            source_ref="local:vllm",
        )
        tracker = ComponentTracker(
            component_id="vllm",
            display_name="vLLM",
            category="runtime",
            local_source=StaticVersionSource(
                source_kind="local",
                source_name="static-local",
                source_ref="local:vllm",
                snapshot=local_snapshot,
            ),
            upstream_source=GitHubReleaseVersionSource(
                repository="vllm-project/vllm",
                source_ref="https://github.com/vllm-project/vllm/releases/latest",
            ),
        )
        monitor = UpdateMonitor(trackers=(tracker,), repo_root=REPO_ROOT)

        report = monitor.generate_report(allow_network=False, timeout_seconds=1.0)

        self.assertEqual(report.components[0].status, "unavailable")
        self.assertIsNone(report.components[0].update_available)
        self.assertIsNotNone(report.components[0].latest)
        assert report.components[0].latest is not None
        self.assertEqual(report.components[0].latest.status, "unavailable")
        self.assertEqual(report.components[0].latest.error, "network disabled")

        payload = report.to_dict()
        round_trip = UpdateReport.from_dict(payload)
        self.assertEqual(
            json.dumps(payload, sort_keys=True),
            json.dumps(round_trip.to_dict(), sort_keys=True),
        )

        rendered = render_update_report_text(report)
        self.assertIn("network=off", rendered)
        self.assertIn("vLLM [runtime]", rendered)
        self.assertIn("status=unavailable", rendered)

    def test_developer_update_report_cli_wires_json_and_text_output(self) -> None:
        current = VersionSnapshot(
            source_kind="local",
            source_name="static-local",
            status="available",
            raw_version="0.1.0",
            normalized_version="0.1.0",
            source_ref="local:demo",
        )
        latest = VersionSnapshot(
            source_kind="upstream",
            source_name="static-upstream",
            status="available",
            raw_version="0.2.0",
            normalized_version="0.2.0",
            source_ref="upstream:demo",
            summary="Adds the thing we want.",
        )
        report = UpdateReport(
            generated_at="2026-04-10T10:00:00Z",
            allow_network=False,
            timeout_seconds=2.5,
            repo_root=str(REPO_ROOT),
            components=(
                ComponentUpdateReport(
                    component_id="demo",
                    display_name="Demo Component",
                    category="runtime",
                    current=current,
                    latest=latest,
                    update_available=True,
                    status="update-available",
                    change_summary=latest.summary,
                    source_refs=("local:demo", "upstream:demo"),
                    notes=("offline test",),
                ),
            ),
        )

        class FakeMonitor:
            def __init__(self) -> None:
                self.calls: list[tuple[bool, float]] = []

            def generate_report(self, *, allow_network: bool, timeout_seconds: float) -> UpdateReport:
                self.calls.append((allow_network, timeout_seconds))
                return report

        fake_monitor = FakeMonitor()

        json_buffer = io.StringIO()
        with patch(
            "cluster.orchestrator.clusterctl.build_default_update_monitor",
            return_value=fake_monitor,
        ), redirect_stdout(json_buffer):
            exit_code_json = clusterctl_main(
                [
                    "developer-update-report",
                    "--repo-root",
                    str(REPO_ROOT),
                    "--offline",
                    "--timeout-seconds",
                    "2.5",
                ]
            )

        text_buffer = io.StringIO()
        with patch(
            "cluster.orchestrator.clusterctl.build_default_update_monitor",
            return_value=fake_monitor,
        ), redirect_stdout(text_buffer):
            exit_code_text = clusterctl_main(
                [
                    "developer-update-report",
                    "--repo-root",
                    str(REPO_ROOT),
                    "--format",
                    "text",
                    "--timeout-seconds",
                    "7",
                ]
            )

        json_payload = json.loads(json_buffer.getvalue())
        text_output = text_buffer.getvalue()

        self.assertEqual(exit_code_json, 0)
        self.assertEqual(exit_code_text, 0)
        self.assertEqual(fake_monitor.calls, [(False, 2.5), (True, 7.0)])
        self.assertEqual(json_payload["summary"]["update_available"], 1)
        self.assertEqual(json_payload["components"][0]["status"], "update-available")
        self.assertIn("Update report generated_at=", text_output)
        self.assertIn("Demo Component", text_output)
