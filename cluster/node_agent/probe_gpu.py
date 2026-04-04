from __future__ import annotations

import argparse
import csv
from datetime import timedelta
import json
import shutil
import socket
import subprocess
import sys
from typing import Iterable

from cluster.models import GPUInventory, LeaseInfo, NodeInventory, parse_datetime, utc_now


class ProbeError(RuntimeError):
    """Raised when the local GPU probe cannot be completed."""


NVIDIA_SMI_QUERY = ",".join(
    [
        "index",
        "uuid",
        "memory.total",
        "memory.free",
        "utilization.gpu",
    ]
)


def _normalize_optional(value: str) -> str | None:
    normalized = value.strip()
    if normalized in {"", "N/A", "[Not Supported]"}:
        return None
    return normalized


def _normalize_optional_int(value: str) -> int | None:
    normalized = _normalize_optional(value)
    return int(normalized) if normalized is not None else None


def probe_gpus(nvidia_smi_bin: str = "nvidia-smi", allow_empty: bool = True) -> list[GPUInventory]:
    if shutil.which(nvidia_smi_bin) is None:
        if allow_empty:
            return []
        raise ProbeError("nvidia-smi is not available on PATH")

    cmd = [
        nvidia_smi_bin,
        f"--query-gpu={NVIDIA_SMI_QUERY}",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        combined = "\n".join(part for part in [stdout, stderr] if part)
        if allow_empty and "No devices were found" in combined:
            return []
        raise ProbeError(f"nvidia-smi probe failed: {combined or exc}") from exc

    rows = [row for row in csv.reader(result.stdout.splitlines()) if row]
    gpus: list[GPUInventory] = []
    for row in rows:
        if len(row) < 5:
            raise ProbeError(f"Unexpected nvidia-smi row: {row!r}")
        gpus.append(
            GPUInventory(
                index=int(row[0].strip()),
                uuid=_normalize_optional(row[1]),
                total_memory_mib=int(row[2].strip()),
                free_memory_mib=int(row[3].strip()),
                utilization_pct=_normalize_optional_int(row[4]),
            )
        )
    return gpus


def build_local_inventory(
    *,
    node_id: str,
    host: str | None = None,
    available_until=None,
    lease_duration_seconds: int | None = None,
    cached_models: Iterable[str] = (),
    labels: dict[str, str] | None = None,
    trust_tier: str | None = None,
    network_tier: str | None = None,
    allow_empty: bool = True,
) -> NodeInventory:
    if available_until is None:
        duration = lease_duration_seconds if lease_duration_seconds is not None else 3600
        available_until = utc_now() + timedelta(seconds=duration)

    gpus = tuple(probe_gpus(allow_empty=allow_empty))
    return NodeInventory(
        node_id=node_id,
        host=host or socket.gethostname(),
        lease=LeaseInfo(
            available_until=available_until,
            lease_duration_seconds=lease_duration_seconds,
            source="node-agent-probe",
        ),
        gpus=gpus,
        cached_models=tuple(sorted(set(cached_models))),
        labels=tuple(sorted((labels or {}).items())),
        trust_tier=trust_tier,
        network_tier=network_tier,
    )


def parse_label_items(values: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected label in key=value form, got: {item}")
        key, value = item.split("=", 1)
        labels[key.strip()] = value.strip()
    return labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe local GPUs into a NodeInventory.")
    parser.add_argument("--node-id", default="local-node", help="Stable node identifier.")
    parser.add_argument("--host", default=socket.gethostname(), help="Advertised host value.")
    parser.add_argument(
        "--available-until",
        help="Absolute availability timestamp in ISO-8601 with timezone.",
    )
    parser.add_argument(
        "--lease-duration-seconds",
        type=int,
        default=3600,
        help="Relative lease duration when --available-until is omitted.",
    )
    parser.add_argument(
        "--cached-model",
        action="append",
        default=[],
        help="Repeat to describe cached models on the node.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Repeat key=value labels to attach to the node.",
    )
    parser.add_argument("--trust-tier", help="Optional trust classification.")
    parser.add_argument("--network-tier", help="Optional network classification.")
    parser.add_argument(
        "--no-allow-empty",
        action="store_true",
        help="Fail instead of returning an empty GPU list when no GPU is available.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        available_until = None if args.available_until is None else parse_datetime(args.available_until)
        inventory = build_local_inventory(
            node_id=args.node_id,
            host=args.host,
            available_until=available_until,
            lease_duration_seconds=args.lease_duration_seconds,
            cached_models=args.cached_model,
            labels=parse_label_items(args.label),
            trust_tier=args.trust_tier,
            network_tier=args.network_tier,
            allow_empty=not args.no_allow_empty,
        )
    except (ProbeError, ValueError) as exc:
        print(f"probe error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(inventory.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
