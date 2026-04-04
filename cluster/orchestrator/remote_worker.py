from __future__ import annotations

import argparse
import json
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2 placeholder worker entrypoint.")
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--runtime-class", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--node-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    print(
        json.dumps(
            {
                "status": "worker-started",
                "agent_id": args.agent_id,
                "model": args.model,
                "backend": args.backend,
                "runtime_class": args.runtime_class,
                "gpu_index": args.gpu_index,
                "node_id": args.node_id,
                "single_gpu_only": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
