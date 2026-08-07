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
        self.vision_calls = 0  # real (non-cached) vision API calls this run

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
                if e.code == 429 and _is_daily_quota_error(body):
                    # A daily quota won't recover within a retry backoff window —
                    # hammering it just burns more of tomorrow's calls too.
                    raise LLMError(
                        "Gemini daily free-tier quota exhausted (HTTP 429, RESOURCE_EXHAUSTED, "
                        "a *PerDay* limit). Retrying will not help: wait for the quota to reset "
                        "(daily) or enable billing on the project to raise the limit. "
                        f"Details: {_safe_excerpt(body)}"
                    ) from None
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

    def _hash(self, key: dict) -> str:
        return hashlib.sha256(
            json.dumps(key, sort_keys=True, ensure_ascii=False, default=str).encode()
        ).hexdigest()[:24]

    def _cached(self, key: dict, fn, bypass_read: bool = False, on_hit=None):
        """Cache by a key that pins provider/model/base_url plus the call's own
        content (system+user+schema+name, or prompt+image). A prior build of
        this client hashed a narrower key (no base_url, no call name); we look
        there too and migrate a hit forward, so a cache already paid for in API
        calls doesn't go to waste just because the key got stricter.

        `bypass_read`, used only by vision() under AB_BYPASS_VISION_CACHE,
        skips the read-side lookup (both current and legacy) so the real API
        is always called, while still writing the result to the normal cache
        path afterwards — json_call never sets this, so its caching (and the
        transaction classifier's, which is built on json_call) is untouched.
        """
        full_key = {"provider": self.cfg.provider, "model": self.cfg.model,
                     "base_url": self.cfg.base_url, **key}
        path = os.path.join(self.cache_dir, self._hash(full_key) + ".json")

        if not bypass_read:
            if os.path.exists(path):
                val = json.load(open(path, encoding="utf-8"))
                if on_hit:
                    on_hit()
                return val

            legacy_key = _legacy_cache_key(self.cfg, key)
            if legacy_key is not None:
                legacy_path = os.path.join(self.cache_dir, self._hash(legacy_key) + ".json")
                if os.path.exists(legacy_path):
                    val = json.load(open(legacy_path, encoding="utf-8"))
                    json.dump(val, open(path, "w", encoding="utf-8"), ensure_ascii=False)
                    if on_hit:
                        on_hit()
                    return val

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

        return self._cached({"system": system, "user": user, "schema": schema, "name": name}, run)

    def vision(self, prompt: str, image_path: str) -> str:
        media = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        b64 = base64.b64encode(open(image_path, "rb").read()).decode()
        bypass = os.environ.get("AB_BYPASS_VISION_CACHE") == "1"

        def run():
            # Must print immediately before the real HTTP request, and only
            # here — never on a cache-hit path.
            print(f"VISION_API_CALL file={image_path}")
            self.calls += 1
            self.vision_calls += 1
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

        def on_hit():
            print(f"VISION_CACHE_HIT file={image_path}")

        return self._cached(
            {"prompt": prompt, "image": hashlib.md5(b64.encode()).hexdigest()},
            run, bypass_read=bypass, on_hit=on_hit)


def _legacy_cache_key(cfg: LLMConfig, key: dict) -> dict | None:
    """Reconstruct the pre-fix cache key shape for one call, if it matches
    a known shape, so already-cached entries stay reachable. Returns None for
    key shapes that didn't exist under the old format (e.g. the transaction
    classifier's wire format changed, so there is nothing valid to recover)."""
    if {"system", "user", "schema", "name"} <= key.keys():
        return {"s": key["system"], "u": key["user"], "sc": key["schema"],
                "provider": cfg.provider, "m": cfg.model}
    if {"prompt", "image"} <= key.keys():
        return {"p": key["prompt"], "img": key["image"],
                "provider": cfg.provider, "m": cfg.model}
    return None


def _is_daily_quota_error(body: str) -> bool:
    low = body.lower().replace(" ", "")
    return "resource_exhausted" in low and "perday" in low


def _safe_excerpt(body: str, limit: int = 400) -> str:
    """First `limit` chars of an error body, with anything key-shaped scrubbed."""
    import re
    scrubbed = re.sub(r'"?key"?\s*[:=]\s*"?[A-Za-z0-9_\-]{16,}"?', '"key": "<redacted>"', body,
                       flags=re.IGNORECASE)
    return scrubbed[:limit]
