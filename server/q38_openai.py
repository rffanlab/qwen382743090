#!/usr/bin/env python3
"""Minimal OpenAI-compatible HTTP surface for Q38RT.

Native generation is intentionally gated until the SM86 decode path is correct.
A reference upstream can be configured to exercise clients and compare outputs
against llama.cpp or another OpenAI-compatible oracle.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib import request, error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from q38pack_format import PackReader  # noqa: E402


def send_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def openai_error(message: str, code: str, err_type: str = "server_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": err_type, "param": None, "code": code}}


class State:
    def __init__(self, model_path: Path, model_id: str, reference_base_url: str | None, reference_model: str | None):
        self.model_path = model_path
        self.model_id = model_id
        self.reference_base_url = reference_base_url.rstrip("/") if reference_base_url else None
        self.reference_model = reference_model
        with PackReader(model_path) as pack:
            self.tensor_count = pack.header.tensor_count
            self.architecture = pack.manifest.get("model", {}).get("architecture")
        self.native_ready = False

    @property
    def generation_ready(self) -> bool:
        return self.native_ready or self.reference_base_url is not None


class Handler(BaseHTTPRequestHandler):
    server_version = "q38rt/0.1"

    @property
    def state(self) -> State:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:
        if self.path == "/health":
            send_json(self, 200, {
                "status": "ok",
                "native_decode_ready": self.state.native_ready,
                "reference_backend": bool(self.state.reference_base_url),
                "model": self.state.model_id,
                "architecture": self.state.architecture,
                "tensor_count": self.state.tensor_count,
            })
            return

        if self.path == "/v1/models":
            now = int(time.time())
            send_json(self, 200, {
                "object": "list",
                "data": [{
                    "id": self.state.model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": "q38rt",
                }],
            })
            return

        send_json(self, 404, openai_error("Not found", "not_found", "invalid_request_error"))

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            send_json(self, 404, openai_error("Not found", "not_found", "invalid_request_error"))
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
        except Exception as exc:
            send_json(self, 400, openai_error(f"Invalid JSON request: {exc}", "invalid_json", "invalid_request_error"))
            return

        if self.state.reference_base_url:
            self.proxy_reference(payload)
            return

        send_json(self, 503, openai_error(
            "Q38 native decode is not implemented in this build yet. "
            "Use --reference-base-url for correctness plumbing while native SM86 kernels are under development.",
            "native_decode_not_ready",
        ))

    def proxy_reference(self, payload: dict[str, Any]) -> None:
        assert self.state.reference_base_url
        forwarded = dict(payload)
        if self.state.reference_model:
            forwarded["model"] = self.state.reference_model
        url = self.state.reference_base_url + "/v1/chat/completions"
        raw = json.dumps(forwarded, ensure_ascii=False).encode("utf-8")
        req = request.Request(url, data=raw, method="POST", headers={"Content-Type": "application/json"})
        auth = self.headers.get("Authorization")
        if auth:
            req.add_header("Authorization", auth)

        try:
            upstream = request.urlopen(req, timeout=600)
        except error.HTTPError as exc:
            body = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        except Exception as exc:
            send_json(self, 502, openai_error(f"Reference backend failed: {exc}", "reference_backend_error"))
            return

        content_type = upstream.headers.get("Content-Type", "application/json")
        is_stream = bool(payload.get("stream"))
        if not is_stream:
            body = upstream.read()
            self.send_response(upstream.status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(upstream.status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        while True:
            chunk = upstream.read(8192)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description="Q38RT OpenAI-compatible server")
    ap.add_argument("--model", type=Path, required=True, help="Q38PACK model")
    ap.add_argument("--model-id", default="qwen3.8-27b-q38")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reference-base-url", help="temporary OpenAI-compatible oracle, e.g. http://127.0.0.1:8080")
    ap.add_argument("--reference-model", help="model name sent to the reference backend")
    args = ap.parse_args()

    state = State(args.model, args.model_id, args.reference_base_url, args.reference_model)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = state  # type: ignore[attr-defined]
    print(f"q38rt serving http://{args.host}:{args.port} model={args.model_id}")
    print(f"native_decode_ready={state.native_ready} reference_backend={bool(state.reference_base_url)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
