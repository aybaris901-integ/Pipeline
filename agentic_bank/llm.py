"""Thin Gemini REST client: cached, retried, and forced to emit valid JSON.

Structured output is obtained with Gemini's `responseSchema` / JSON response
mode, which is far more reliable than asking for JSON in prose and stripping
code fences. Only the standard-library `urllib` is used to talk to the API.
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
import time
import urllib.request
import urllib.error

from .config import LLMConfig


class LLMError(RuntimeError):
    pass


# JSON-Schema (as used by this project's schemas) -> Gemini Schema subset.
_TYPE_MAP = {
    "string": "STRING", "number": "NUMBER", "integer": "INTEGER",
    "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT",
}


def _to_gemini_schema(node):
    """Recursively convert a JSON-Schema dict into Gemini's Schema subset.

    Gemini's Schema has no union `type` list — a `["string", "null"]` type
    becomes `type: STRING, nullable: true`. `enum` is only valid for STRING
    types in the Gemini schema, so non-string enums are dropped (the prompt
    and downstream validation still constrain the value).
    """
    if not isinstance(node, dict):
        return node
    out: dict = {}
    t = node.get("type")
    nullable = False
    if isinstance(t, list):
        types = [x for x in t if x is not None and x != "null"]
        nullable = any(x is None or x == "null" for x in t)
        t = types[0] if types else None
    if t in _TYPE_MAP:
        out["type"] = _TYPE_MAP[t]
    if nullable:
        out["nullable"] = True
    if "description" in node:
        out["description"] = node["description"]
    if "enum" in node and out.get("type") == "STRING":
        vals = [v for v in node["enum"] if isinstance(v, str)]
        if vals:
            out["enum"] = vals
    if "properties" in node:
        out["properties"] = {k: _to_gemini_schema(v) for k, v in node["properties"].items()}
    if "required" in node:
        out["required"] = node["required"]
    if "items" in node:
        out["items"] = _to_gemini_schema(node["items"])
    return out


class LLM:
    def __init__(self, cfg: LLMConfig, cache_dir: str = ".cache/llm"):
        self.cfg = cfg
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.calls = 0

    # ------------------------------------------------------------ plumbing --
    def _post(self, payload: dict) -> dict:
        url = f"{self.cfg.base_url}/v1beta/models/{self.cfg.model}:generateContent"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-goog-api-key": self.cfg.api_key,
            },
        )
        delay = 2.0
        for attempt in range(self.cfg.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                if e.code in (429, 500, 502, 503, 504) and attempt < self.cfg.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError(f"Gemini API error {e.code}: {_safe_excerpt(body)}") from None
            except urllib.error.URLError:
                if attempt < self.cfg.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise LLMError("Gemini API request failed: network error") from None
        raise LLMError("exhausted retries")

    def _cached(self, key: dict, fn):
        full_key = {**key, "provider": self.cfg.provider, "m": self.cfg.model}
        h = hashlib.sha256(json.dumps(full_key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
        path = os.path.join(self.cache_dir, h + ".json")
        if os.path.exists(path):
            return json.load(open(path, encoding="utf-8"))
        val = fn()
        json.dump(val, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        return val

    @staticmethod
    def _extract_text(data: dict) -> str:
        feedback = data.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise LLMError(f"Gemini blocked the prompt: {feedback.get('blockReason')}")
        candidates = data.get("candidates") or []
        if not candidates:
            raise LLMError("Gemini returned no candidates (empty or blocked response)")
        cand = candidates[0]
        finish = cand.get("finishReason")
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)
        if not text.strip():
            raise LLMError(f"Gemini returned empty output (finishReason={finish!r})")
        return text

    # --------------------------------------------------------------- public --
    def json_call(self, system: str, user: str, schema: dict, name: str = "emit") -> dict:
        """Return a dict validated against `schema` by Gemini's structured output."""
        def run():
            self.calls += 1
            payload = {
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "systemInstruction": {"parts": [{"text": system}]},
                "generationConfig": {
                    "temperature": self.cfg.temperature,
                    "maxOutputTokens": self.cfg.max_tokens,
                    "responseMimeType": "application/json",
                    "responseSchema": _to_gemini_schema(schema),
                },
            }
            data = self._post(payload)
            text = self._extract_text(data)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                raise LLMError(f"Gemini returned malformed JSON: {e}") from None
            if not isinstance(parsed, dict):
                raise LLMError(f"Gemini returned a JSON {type(parsed).__name__}, expected an object")
            return parsed

        return self._cached({"s": system, "u": user, "sc": schema}, run)

    def vision(self, prompt: str, image_path: str) -> str:
        media = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        b64 = base64.b64encode(open(image_path, "rb").read()).decode()

        def run():
            print(f"VISION_API_CALL file={image_path}")
            self.calls += 1
            payload = {
                "contents": [{"role": "user", "parts": [
                    {"inlineData": {"mimeType": media, "data": b64}},
                    {"text": prompt},
                ]}],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": self.cfg.max_tokens,
                },
            }
            data = self._post(payload)
            return self._extract_text(data)

        return self._cached(
            {"p": prompt, "img": hashlib.md5(b64.encode()).hexdigest()}, run)


def _safe_excerpt(body: str, limit: int = 400) -> str:
    """First `limit` chars of an error body, with anything key-shaped scrubbed."""
    import re
    scrubbed = re.sub(r'"?key"?\s*[:=]\s*"?[A-Za-z0-9_\-]{16,}"?', '"key": "<redacted>"', body,
                       flags=re.IGNORECASE)
    return scrubbed[:limit]
