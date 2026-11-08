"""Helpers for the LLM server (llama.cpp `llama-server`, OpenAI-compatible API).

In Docker and on the servers the LLM runs in its own container. Locally on the Mac it
runs natively (Metal GPU) with start_llm_server.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import httpx


def start_llm_server(
    gguf_path: str | Path,
    alias: str,
    port: int = 8001,
    ctx_size: int = 24576,
    parallel: int = 4,
    log_path: str | Path | None = None,
) -> subprocess.Popen:
    """Launch llama-server in the background (all layers on the GPU when there is one)."""
    cmd = [
        "llama-server", "-m", str(gguf_path), "--alias", alias,
        "--host", "127.0.0.1", "--port", str(port),
        "-c", str(ctx_size), "--parallel", str(parallel), "-ngl", "99",
    ]
    log = open(log_path, "w") if log_path else subprocess.DEVNULL
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)


def health_url(api_base: str) -> str:
    """llama-server exposes /health next to /v1."""
    return api_base.rstrip("/").removesuffix("/v1") + "/health"


def is_llm_ready(api_base: str, timeout: float = 3.0) -> bool:
    try:
        return httpx.get(health_url(api_base), timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


def wait_until_ready(api_base: str, timeout_s: float = 600, poll_s: float = 3) -> None:
    """Block until the server answers /health (the model is loaded)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if is_llm_ready(api_base):
            return
        time.sleep(poll_s)
    raise TimeoutError(f"LLM server at {api_base} not ready after {timeout_s}s")


def stop_llm_server(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()


def chat_completion(
    api_base: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 256,
    timeout: float = 900,
) -> str:
    """One call to /v1/chat/completions; returns the text of the answer."""
    resp = httpx.post(
        api_base.rstrip("/") + "/chat/completions",
        json={"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"] or ""
