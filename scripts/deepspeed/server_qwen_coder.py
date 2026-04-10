from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal DeepSpeed-backed OpenAI-compatible server for Qwen coder profiles."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int)
    parser.add_argument("--kernel-inject", action="store_true")
    parser.add_argument("--enable-cuda-graph", action="store_true")
    parser.add_argument("--use-triton", action="store_true")
    parser.add_argument("--triton-autotune", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser


def _require_dependency(name: str):
    try:
        return __import__(name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing required dependency {name!r}. Install the runtime with "
            "scripts/runtime/install_deepspeed.sh before launching this server."
        ) from exc


class _DeepSpeedRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        torch = _require_dependency("torch")
        deepspeed = _require_dependency("deepspeed")
        transformers = _require_dependency("transformers")

        self._dist = torch.distributed
        self._lock = threading.Lock()
        self.model_id = args.model
        self.max_model_len = args.max_model_len
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.rank = int(os.environ.get("RANK", str(self.local_rank)))
        self.world_size = int(os.environ.get("WORLD_SIZE", str(args.tensor_parallel_size)))

        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

        if self.world_size > 1 and not self._dist.is_initialized():
            deepspeed.init_distributed(dist_backend="nccl")

        dtype_map = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(args.dtype.lower())
        if torch_dtype is None:
            raise RuntimeError(f"Unsupported dtype for DeepSpeed server: {args.dtype!r}")

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_source = args.checkpoint_dir or args.model
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        model.eval()

        kwargs: dict[str, Any] = {
            "mp_size": self.world_size,
            "dtype": torch_dtype,
            "replace_with_kernel_inject": args.kernel_inject,
        }
        if args.use_triton:
            kwargs["use_triton"] = True
        if args.triton_autotune:
            kwargs["triton_autotune"] = True
        if args.enable_cuda_graph:
            kwargs["enable_cuda_graph"] = True

        try:
            engine = deepspeed.init_inference(model, **kwargs)
        except TypeError:
            engine = deepspeed.init_inference(
                model,
                mp_size=self.world_size,
                dtype=torch_dtype,
                replace_with_kernel_inject=args.kernel_inject,
            )

        self.tokenizer = tokenizer
        self.module = getattr(engine, "module", engine)

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def _broadcast(self, payload: dict[str, Any]) -> None:
        if self.world_size <= 1:
            return
        data = [payload]
        self._dist.broadcast_object_list(data, src=0)

    def _await_command(self) -> dict[str, Any]:
        data: list[dict[str, Any] | None] = [None]
        self._dist.broadcast_object_list(data, src=0)
        command = data[0]
        if command is None:
            raise RuntimeError("DeepSpeed worker received an empty command payload")
        return command

    def _build_prompt(self, payload: dict[str, Any]) -> str:
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            prompt_lines: list[str] = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", "user")).strip() or "user"
                content = message.get("content", "")
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(str(item.get("text", "")))
                    content = "".join(text_parts)
                prompt_lines.append(f"{role}: {content}")
            if prompt_lines:
                prompt_lines.append("assistant:")
                return "\n".join(prompt_lines)
        prompt = payload.get("prompt")
        if prompt is None:
            raise RuntimeError("DeepSpeed request must include either messages or prompt")
        return str(prompt)

    def _generate_impl(self, payload: dict[str, Any]) -> dict[str, Any]:
        prompt = self._build_prompt(payload)
        max_new_tokens = int(payload.get("max_tokens") or payload.get("max_new_tokens") or 128)
        temperature = float(payload.get("temperature", 0.0) or 0.0)
        top_p = float(payload.get("top_p", 1.0) or 1.0)
        do_sample = temperature > 0.0

        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.local_rank)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.local_rank)

        if self.max_model_len is not None and input_ids.shape[-1] > self.max_model_len:
            input_ids = input_ids[:, -self.max_model_len :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_model_len :]

        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if attention_mask is not None:
            generate_kwargs["attention_mask"] = attention_mask
        if do_sample:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = max(temperature, 1e-5)
            generate_kwargs["top_p"] = top_p

        generated = self.module.generate(**generate_kwargs)
        new_tokens = generated[0][input_ids.shape[-1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        prompt_tokens = int(input_ids.shape[-1])
        completion_tokens = int(new_tokens.shape[-1])
        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.world_size > 1 and self.is_primary:
                self._broadcast({"type": "generate", "payload": payload})
            result = self._generate_impl(payload)
            if self.world_size > 1:
                self._dist.barrier()
            return result

    def worker_loop(self) -> None:
        if self.is_primary:
            raise RuntimeError("worker_loop should run only on non-primary DeepSpeed ranks")
        while True:
            command = self._await_command()
            command_type = command.get("type")
            if command_type == "shutdown":
                if self.world_size > 1:
                    self._dist.barrier()
                return
            if command_type != "generate":
                raise RuntimeError(f"Unsupported DeepSpeed worker command: {command_type!r}")
            self._generate_impl(dict(command.get("payload") or {}))
            if self.world_size > 1:
                self._dist.barrier()

    def shutdown(self) -> None:
        if self.world_size > 1 and self.is_primary:
            self._broadcast({"type": "shutdown"})
            self._dist.barrier()


class _Handler(BaseHTTPRequestHandler):
    server: "_DeepSpeedHTTPServer"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json_response(HTTPStatus.OK, {"status": "ok", "model": self.server.runtime.model_id})
            return
        if self.path == "/v1/models":
            self._json_response(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.runtime.model_id,
                            "object": "model",
                            "owned_by": "deepspeed",
                        }
                    ],
                },
            )
            return
        self._json_response(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or "0")
        payload = {}
        if length:
            payload = json.loads(self.rfile.read(length))
        if self.path == "/v1/chat/completions":
            result = self.server.runtime.generate(payload)
            response = {
                "id": "chatcmpl-deepspeed",
                "object": "chat.completion",
                "model": self.server.runtime.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result["text"]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["total_tokens"],
                },
            }
            self._json_response(HTTPStatus.OK, response)
            return
        if self.path == "/v1/completions":
            result = self.server.runtime.generate(payload)
            response = {
                "id": "cmpl-deepspeed",
                "object": "text_completion",
                "model": self.server.runtime.model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": result["text"],
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": result["completion_tokens"],
                    "total_tokens": result["total_tokens"],
                },
            }
            self._json_response(HTTPStatus.OK, response)
            return
        self._json_response(HTTPStatus.NOT_FOUND, {"error": {"message": "Not found"}})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _json_response(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _DeepSpeedHTTPServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], runtime: _DeepSpeedRuntime) -> None:
        super().__init__(address, _Handler)
        self.runtime = runtime


def main() -> int:
    args, _unknown = _build_parser().parse_known_args()
    runtime = _DeepSpeedRuntime(args)
    if not runtime.is_primary:
        runtime.worker_loop()
        return 0

    server = _DeepSpeedHTTPServer((args.host, args.port), runtime)
    try:
        server.serve_forever()
    finally:
        runtime.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
