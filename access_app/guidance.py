"""Build the spoken retail line: template or external LLM (Ollama / OpenAI-compatible)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from access_app.config import (
    GUIDANCE_HEADER,
    GUIDANCE_MODE,
    GUIDANCE_TEMPERATURE,
    LLM_API,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TIMEOUT_S,
)
from retail_guide import guidance_line


def build_guidance(
    objects: list[tuple[str, str, str]],
    person: str | None = None,
) -> str:
    """
    objects: [(name, shelf, description), ...]
    person: matched face name, or None
    """
    pairs = [(name, shelf) for name, shelf, _desc in objects]
    template = guidance_line(pairs, person)
    if not template:
        return ""
    if GUIDANCE_MODE != "llm":
        return template
    try:
        spoken = _external_llm(objects, person)
        return spoken or template
    except Exception as exc:  # noqa: BLE001
        print(f"[guidance] external LLM skipped, using template: {exc}")
        return template


def _facts_blob(
    objects: list[tuple[str, str, str]],
    person: str | None,
) -> str:
    lines: list[str] = []
    if person and person != "Unknown":
        lines.append(f"person: {person}")
    for name, shelf, description in objects:
        bit = f"item: {name}"
        if shelf:
            bit += f"; shelf: {shelf}"
        if description:
            bit += f"; description: {description}"
        lines.append(bit)
    return "\n".join(lines)


def _external_llm(
    objects: list[tuple[str, str, str]],
    person: str | None,
) -> str:
    """Rewrite known facts with an external LLM. Facts are passed in the prompt only."""
    facts = _facts_blob(objects, person)
    system = GUIDANCE_HEADER or (
        "You are a store guide. Reply with exactly one short spoken sentence. "
        "Use only the facts in the user message. Do not invent items or shelves. "
        "No quotes, no markdown, no lists."
    )
    user = (
        "Speak one short retail guidance sentence for a shopper. "
        "Use only these facts:\n"
        f"{facts}"
    )
    if LLM_API == "openai":
        return _openai_chat(system, user)
    return _ollama_chat(system, user)


def _ollama_chat(system: str, user: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "stream": False,
        "options": {"temperature": GUIDANCE_TEMPERATURE},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    raw = _post_json(f"{LLM_BASE_URL}/api/chat", payload)
    message = raw.get("message") if isinstance(raw, dict) else None
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return " ".join(content.strip().split())
    # Older generate-style responses
    for key in ("response", "answer", "text"):
        value = raw.get(key) if isinstance(raw, dict) else None
        if isinstance(value, str) and value.strip():
            return " ".join(value.strip().split())
    return ""


def _openai_chat(system: str, user: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "temperature": GUIDANCE_TEMPERATURE,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    raw = _post_json(f"{LLM_BASE_URL}/v1/chat/completions", payload)
    choices = raw.get("choices") if isinstance(raw, dict) else None
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return " ".join(content.strip().split())
    return ""


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=LLM_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{exc.code} {url}: {detail[:200]}") from exc
