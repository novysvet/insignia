#!/usr/bin/env python3
"""Insignia Web UI and OpenAI-compatible API for the local WSL engine."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


MODEL_ID = "glm-5.3-flash-nvfp4"
ROOT = Path(__file__).resolve().parent
TOKENIZER_PATH = os.environ.get(
    "INSIGNIA_WEB_TOKENIZER",
    r"\\wsl.localhost\Arch\var\lib\insignia\glm53-flash-text\tokenizer.json",
)
ENGINE = os.environ.get("INSIGNIA_WEB_ENGINE", "/var/tmp/insignia-build/glm53-generate")
MODEL_ROOT = os.environ.get("INSIGNIA_WEB_MODEL_ROOT", "/var/lib/insignia/glm53-flash-text")
MODEL_INDEX = os.environ.get("INSIGNIA_WEB_MODEL_INDEX", "/var/lib/insignia/glm53-flash-text.index")
FP8_PREFIX = os.environ.get("INSIGNIA_WEB_FP8_PREFIX", "/var/lib/insignia/glm53-fp8-g64")
MAX_TOKENS = int(os.environ.get("INSIGNIA_WEB_MAX_TOKENS", "512"))
SPEED_PROFILE = os.environ.get("INSIGNIA_WEB_SPEED_PROFILE", "top6-cache").lower()
TOKENIZER = Tokenizer.from_file(TOKENIZER_PATH)
ENGINE_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()
STATE: dict[str, Any] = {"busy": False, "started_at": None, "request_id": None}
EOS_IDS = {value for value in (TOKENIZER.token_to_id("<|im_end|>"), 151643) if value is not None}


def engine_assignments(profile: str, cache_mb: str | None = None) -> list[str]:
    assignments = [
        "INSIGNIA_GLM53_DFLASH2=1",
        "INSIGNIA_GLM53_DFLASH2_FP8=/var/lib/insignia/glm53-dflash2-fp8-fixed",
        f"INSIGNIA_GLM53_READERS={os.environ.get('INSIGNIA_WEB_READERS', '4')}",
    ]
    if profile == "top6-cache":
        assignments += [
            "INSIGNIA_GLM53_DF_APPROX_TOPM=6",
            "INSIGNIA_GLM53_DF_CACHE_ROUTE_K=32",
            "INSIGNIA_GLM53_DF_CACHE_ROUTE_REGRET=.0010",
            "INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=8",
        ]
    elif profile != "exact":
        raise ValueError("INSIGNIA_WEB_SPEED_PROFILE must be 'top6-cache' or 'exact'")
    if cache_mb:
        assignments.append(f"INSIGNIA_GLM53_EXPERT_CACHE_MB={int(cache_mb)}")
    return assignments


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") not in ("text", "input_text"):
                raise ValueError("Only text message content is supported")
            pieces.append(str(item.get("text", "")))
        return "\n".join(pieces)
    raise ValueError("message content must be a string or text-content array")


def chat_prompt(messages: list[dict[str, Any]]) -> str:
    if not messages:
        raise ValueError("messages must not be empty")
    parts = ["[gMASK]<sop>"]
    if messages[0].get("role") != "system":
        parts.append("<|system|>Reasoning Effort: Max")
    for message in messages:
        role = message.get("role")
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"unsupported message role: {role!r}")
        parts.append(f"<|{role}|>{message_text(message.get('content', ''))}")
    parts.append("<|assistant|><think>")
    return "".join(parts)


def trim_ids(ids: list[int]) -> list[int]:
    for index, token in enumerate(ids):
        if token in EOS_IDS:
            return ids[:index]
    return ids


def parse_engine_output(output: str) -> tuple[list[int], dict[str, Any]]:
    match = re.search(r"^greedy IDs(?:\s+([0-9 ]+))?\s*$", output, re.MULTILINE)
    if not match:
        tail = "\n".join(output.splitlines()[-12:])
        raise RuntimeError(f"engine produced no committed token stream\n{tail}")
    ids = trim_ids([int(value) for value in (match.group(1) or "").split()])
    timing: dict[str, Any] = {}
    prompt = re.search(r"(\d+)-token prompt ([0-9.]+) s", output)
    decode = re.search(r"([0-9.]+) ms/token", output)
    acceptance = re.search(r"\(([0-9.]+) accepted/round", output)
    if prompt:
        timing.update(prompt_tokens=int(prompt.group(1)), prefill_seconds=float(prompt.group(2)))
    if decode:
        timing["decode_ms_per_token"] = float(decode.group(1))
    if acceptance:
        timing["accepted_per_round"] = float(acceptance.group(1))
    return ids, timing


def run_engine(prompt: str, max_tokens: int, request_id: str) -> tuple[str, list[int], dict[str, Any]]:
    prompt_ids = TOKENIZER.encode(prompt, add_special_tokens=False).ids
    if len(prompt_ids) + max_tokens > 8193:
        raise ValueError("prompt plus max_tokens exceeds the current 8192-token engine context")
    command = [
        "wsl.exe", "-d", "Arch", "--", "env",
        *engine_assignments(SPEED_PROFILE, os.environ.get("INSIGNIA_WEB_EXPERT_CACHE_MB")),
        ENGINE, MODEL_ROOT, MODEL_INDEX, ",".join(map(str, prompt_ids)), "0",
        str(max_tokens), FP8_PREFIX,
    ]
    with ENGINE_LOCK:
        with STATE_LOCK:
            STATE.update(busy=True, started_at=time.time(), request_id=request_id)
        try:
            completed = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", check=False,
            )
        finally:
            with STATE_LOCK:
                STATE.update(busy=False, started_at=None, request_id=None)
    if completed.returncode:
        tail = "\n".join(completed.stdout.splitlines()[-18:])
        raise RuntimeError(f"engine exited with status {completed.returncode}\n{tail}")
    ids, timing = parse_engine_output(completed.stdout)
    timing["prompt_tokens"] = len(prompt_ids)
    timing["completion_tokens"] = len(ids)
    timing["speed_profile"] = SPEED_PROFILE
    return TOKENIZER.decode(ids, skip_special_tokens=True), ids, timing


def openapi() -> dict[str, Any]:
    return {
        "openapi": "3.1.0",
        "info": {"title": "Insignia API", "version": "0.1.0"},
        "servers": [{"url": "/v1"}],
        "paths": {
            "/models": {"get": {"summary": "List models", "responses": {"200": {"description": "OK"}}}},
            "/chat/completions": {"post": {"summary": "Create a chat completion", "responses": {"200": {"description": "Completion or SSE stream"}}}},
            "/completions": {"post": {"summary": "Create a text completion", "responses": {"200": {"description": "Completion or SSE stream"}}}},
        },
        "components": {"schemas": {"ChatCompletionRequest": {
            "type": "object", "required": ["messages"], "properties": {
                "model": {"type": "string", "default": MODEL_ID},
                "messages": {"type": "array", "items": {"type": "object"}},
                "max_tokens": {"type": "integer", "minimum": 1, "maximum": MAX_TOKENS},
                "stream": {"type": "boolean", "default": False},
            }
        }}},
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "Insignia/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "authorization, content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def json_response(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def error_response(self, status: int, message: str, code: str = "invalid_request_error") -> None:
        self.json_response(status, {"error": {"message": message, "type": code, "code": code}})

    def authorized(self) -> bool:
        key = os.environ.get("INSIGNIA_API_KEY")
        return not key or self.headers.get("Authorization") == f"Bearer {key}"

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = (ROOT / "index.html").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/v1/models":
            if not self.authorized():
                return self.error_response(HTTPStatus.UNAUTHORIZED, "Invalid API key", "authentication_error")
            self.json_response(HTTPStatus.OK, {"object": "list", "data": [{
                "id": MODEL_ID, "object": "model", "created": 0, "owned_by": "insignia",
            }]})
        elif self.path == "/healthz":
            with STATE_LOCK:
                state = dict(STATE)
            self.json_response(HTTPStatus.OK, {"status": "busy" if state["busy"] else "ready", "model": MODEL_ID, **state})
        elif self.path == "/openapi.json":
            self.json_response(HTTPStatus.OK, openapi())
        else:
            self.error_response(HTTPStatus.NOT_FOUND, "Route not found", "not_found")

    def do_POST(self) -> None:
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            return self.error_response(HTTPStatus.NOT_FOUND, "Route not found", "not_found")
        if not self.authorized():
            return self.error_response(HTTPStatus.UNAUTHORIZED, "Invalid API key", "authentication_error")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_048_576:
                raise ValueError("request body must be between 1 byte and 1 MiB")
            request = json.loads(self.rfile.read(length))
            max_tokens = int(request.get("max_tokens", request.get("max_completion_tokens", 256)))
            if not 1 <= max_tokens <= MAX_TOKENS:
                raise ValueError(f"max_tokens must be between 1 and {MAX_TOKENS}")
            if self.path.endswith("/chat/completions"):
                messages = request.get("messages")
                if not isinstance(messages, list):
                    raise ValueError("messages must be an array")
                prompt = chat_prompt(messages)
                kind = "chat.completion"
            else:
                raw_prompt = request.get("prompt", "")
                if not isinstance(raw_prompt, str):
                    raise ValueError("only a single string prompt is supported")
                prompt = raw_prompt
                kind = "text_completion"
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            return self.error_response(HTTPStatus.BAD_REQUEST, str(error))

        request_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        stream = bool(request.get("stream", False))
        if stream:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                if kind == "chat.completion":
                    self.sse({"id": request_id, "object": "chat.completion.chunk", "created": created,
                              "model": MODEL_ID, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
                text, ids, timing = run_engine(prompt, max_tokens, request_id)
                if kind == "chat.completion":
                    chunk = {"id": request_id, "object": "chat.completion.chunk", "created": created,
                             "model": MODEL_ID, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}], "insignia": timing}
                    final = {"id": request_id, "object": "chat.completion.chunk", "created": created,
                             "model": MODEL_ID, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                else:
                    chunk = {"id": request_id, "object": "text_completion", "created": created,
                             "model": MODEL_ID, "choices": [{"index": 0, "text": text, "finish_reason": None}], "insignia": timing}
                    final = {"id": request_id, "object": "text_completion", "created": created,
                             "model": MODEL_ID, "choices": [{"index": 0, "text": "", "finish_reason": "stop"}]}
                self.sse(chunk)
                self.sse(final)
                if request.get("stream_options", {}).get("include_usage"):
                    self.sse({"id": request_id, "object": f"{kind}.chunk", "created": created,
                              "model": MODEL_ID, "choices": [], "usage": usage(timing)})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as error:
                try:
                    self.sse({"error": {"message": str(error), "type": "engine_error", "code": "engine_error"}})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    self.close_connection = True
                except (BrokenPipeError, ConnectionResetError):
                    pass
            return

        try:
            text, ids, timing = run_engine(prompt, max_tokens, request_id)
        except ValueError as error:
            return self.error_response(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:
            return self.error_response(HTTPStatus.INTERNAL_SERVER_ERROR, str(error), "engine_error")
        if kind == "chat.completion":
            choices = [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]
        else:
            choices = [{"index": 0, "text": text, "finish_reason": "stop"}]
        self.json_response(HTTPStatus.OK, {"id": request_id, "object": kind, "created": created,
                                          "model": MODEL_ID, "choices": choices,
                                          "usage": usage(timing), "insignia": timing})

    def sse(self, payload: Any) -> None:
        self.wfile.write(b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n")
        self.wfile.flush()


def usage(timing: dict[str, Any]) -> dict[str, int]:
    prompt = int(timing.get("prompt_tokens", 0))
    completion = int(timing.get("completion_tokens", 0))
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def self_test() -> None:
    prompt = chat_prompt([{"role": "user", "content": "2+2?"}])
    assert prompt == "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>2+2?<|assistant|><think>"
    ids, timing = parse_engine_output("greedy IDs 12 13 151643 99\n9-token prompt 1.250 s; 4 greedy tokens in 2 DFLASH2-k4 rounds (2.00 accepted/round, 100.0 ms/token;)\n")
    assert ids == [12, 13] and timing["prefill_seconds"] == 1.25 and timing["decode_ms_per_token"] == 100.0
    fast = engine_assignments("top6-cache")
    assert "INSIGNIA_GLM53_DF_APPROX_TOPM=6" in fast
    assert "INSIGNIA_GLM53_DF_CACHE_JOINT_OPTIONS=8" in fast
    assert not any(value.startswith("INSIGNIA_GLM53_EXPERT_CACHE_MB=") for value in fast)
    exact = engine_assignments("exact", "32768")
    assert "INSIGNIA_GLM53_DF_APPROX_TOPM=6" not in exact
    assert "INSIGNIA_GLM53_EXPERT_CACHE_MB=32768" in exact
    print("web API self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Insignia serving http://{args.host}:{args.port} ({MODEL_ID})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
