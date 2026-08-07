"""LLM client: cached, retried, and forced to emit valid JSON.

Two providers are supported behind the same public interface
(`json_call`, `vision`, `.calls`, `.vision_calls`):

* **groq** (default) — OpenAI-compatible `/chat/completions` API. Text calls
  run a fixed escalation chain (`cfg.model` -> `cfg.fallback_model` ->
  `cfg.strong_fallback_model`), validating every response against the
  caller's JSON Schema in Python, since Groq's own JSON enforcement is
  weaker than Gemini's `responseSchema` (json_object mode has none; the
  gpt-oss models' structured-output mode is attempted first but the
  applica­tion-side check still runs regardless).
* **gemini** — the original REST client, kept working unchanged for anyone
  who still sets LLM_PROVIDER=gemini / has GEMINI_API_KEY. Structured
  output is obtained with Gemini's `responseSchema` / JSON response mode.

Only the standard-library `urllib` is used to talk to either API.
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
import random
import time
import urllib.request
import urllib.error

from .config import LLMConfig


class LLMError(RuntimeError):
    pass


class LLMAuthError(LLMError):
    """Invalid/missing credentials. Never triggers a model fallback."""


class LLMProviderBlockedError(LLMError):
    """The provider's edge/CDN rejected the request before it reached the
    model (e.g. a Cloudflare anti-bot rule) — not an invalid API key. Never
    triggers a model fallback: retrying with a different model changes
    nothing about the edge block."""


class LLMRequestError(LLMError):
    """The request itself was rejected (bad model name, malformed payload,
    schema the API won't accept). Points at a bug/misconfiguration, not a
    model capability gap — never triggers a model fallback."""


class LLMPromptTooLargeError(LLMRequestError):
    """The prompt's input tokens alone are too close to (or over) the
    model's per-minute token budget, even with an already-minimal completion
    allowance — a payload-size problem, not a transient rate limit. Waiting
    and retrying the same request will not help; the input itself (e.g. a
    batch or document slice) needs to shrink."""


class LLMValidationError(LLMError):
    """The model replied but the output failed JSON parsing or schema
    validation. A model capability/quality problem — safe to escalate to
    the next model in the chain."""


class LLMTransientError(LLMError):
    """Rate limiting or a server-side error survived bounded retries. Safe
    to escalate to the next model in the chain."""


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


def _to_strict_openai_schema(node):
    """Derive an OpenAI/Groq strict-mode Structured Outputs schema from the
    project's JSON Schema, WITHOUT mutating the original.

    Strict mode requires every property to be listed in `required` and
    `additionalProperties: false` on every object. Properties this project
    treats as optional are made nullable instead of dropped, so the model
    can still "omit" them by returning null. This is only used to build the
    wire payload — real optionality/required-ness is still enforced by
    `_validate_json_schema` against the ORIGINAL schema.
    """
    if not isinstance(node, dict):
        return node
    out = dict(node)
    if "properties" in out:
        props = out["properties"]
        new_props = {k: _to_strict_openai_schema(v) for k, v in props.items()}
        required = set(out.get("required") or [])
        for k in props:
            if k not in required:
                child = dict(new_props[k])
                t = child.get("type")
                if isinstance(t, list):
                    if "null" not in t:
                        child["type"] = [*t, "null"]
                elif t is not None:
                    child["type"] = [t, "null"]
                new_props[k] = child
        out["properties"] = new_props
        out["required"] = list(props.keys())
        out["additionalProperties"] = False
    if "items" in out:
        out["items"] = _to_strict_openai_schema(out["items"])
    return out


# ------------------------------------------------------ schema validation --
def _json_type_ok(value, t) -> bool:
    if t == "string":
        return isinstance(value, str)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "array":
        return isinstance(value, list)
    if t == "object":
        return isinstance(value, dict)
    if t == "null":
        return value is None
    return True  # unknown type keyword: don't block on it


def _validate_json_schema(value, schema, path: str = "$", errors: list | None = None) -> list[str]:
    """Recursively check `value` against the (subset of) JSON Schema this
    project uses: type (incl. union/nullable), enum, required, properties,
    items. Returns a list of human-readable error strings (empty = valid).

    This exists because Groq's JSON modes don't enforce a schema as
    reliably as Gemini's `responseSchema` did — a model-quality failure
    here is the signal that triggers escalation to the next model.
    """
    if errors is None:
        errors = []
    if not isinstance(schema, dict):
        return errors

    types = schema.get("type")
    if types is not None:
        allowed = types if isinstance(types, list) else [types]
        if not any(_json_type_ok(value, t) for t in allowed):
            errors.append(f"{path}: expected type {allowed}, got {type(value).__name__}")
            return errors  # type mismatch: nothing more to check at this node

    if "enum" in schema and value is not None and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in allowed values {schema['enum']}")

    if isinstance(value, dict):
        props = schema.get("properties") or {}
        required = set(schema.get("required", []))
        for req in required:
            if req not in value:
                errors.append(f"{path}: missing required property {req!r}")
        for k, v in value.items():
            if k in props:
                if v is None and k not in required:
                    # An optional property explicitly set to null is exactly
                    # what _to_strict_openai_schema's wire-format conversion
                    # asks strict-mode models for in place of omitting the
                    # key outright (OpenAI/Groq strict mode can't actually
                    # drop a key) — json_object-mode models converge on the
                    # same null-for-"not applicable" convention unprompted.
                    # Treat it as the omission it represents rather than
                    # validating None against the property's real type.
                    continue
                _validate_json_schema(v, props[k], f"{path}.{k}", errors)

    if isinstance(value, list):
        item_schema = schema.get("items")
        if item_schema:
            for i, v in enumerate(value):
                _validate_json_schema(v, item_schema, f"{path}[{i}]", errors)

    return errors


def _schema_instructions(schema: dict, name: str) -> str:
    return (
        "\n\nRespond with a single JSON object only (no markdown fences, no "
        f"commentary, no explanation) that matches exactly this JSON Schema "
        f"(the object represents '{name}'):\n{json.dumps(schema, ensure_ascii=False)}"
    )


class LLM:
    # gpt-oss models get a native Structured Outputs attempt first; llama
    # is used with plain JSON mode (json_object) since strict schema
    # support there is not something this project relies on.
    STRUCTURED_OUTPUT_MODELS = {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}

    def __init__(self, cfg: LLMConfig, cache_dir: str = ".cache/llm"):
        self.cfg = cfg
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.calls = 0
        self.vision_calls = 0  # real (non-cached) vision API calls this run

    # ------------------------------------------------------------ plumbing --
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

        Callers that attempt several models in a chain (the Groq path) pass
        the ACTUAL model attempted as `key["model"]`, overriding the
        `self.cfg.model` default below — so each model in the chain gets its
        own cache identity, and a cache hit never spends an API call.

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
                elif self.cfg.verbose:
                    print(f"LLM_CACHE_HIT provider={full_key['provider']} "
                          f"model={full_key['model']} name={key.get('name', '')}")
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

    # --------------------------------------------------------------- public --
    def json_call(self, system: str, user: str, schema: dict, name: str = "emit",
                  max_tokens: int | None = None, extra_validate=None) -> dict:
        """Return a dict validated against `schema`.

        `max_tokens` lets the caller reserve only as much completion budget
        as its schema genuinely needs (defaults to `cfg.max_tokens` when
        omitted) — Groq's free-tier TPM limit counts reserved/requested
        output tokens against the same per-minute budget as the prompt, so a
        generic large default starves small calls of headroom for no
        benefit.

        `extra_validate`, if given, is called as `extra_validate(parsed)`
        after generic schema validation passes and must return a list of
        error strings (empty = OK). It exists for business-rule checks the
        generic JSON-Schema subset can't express — e.g. a field that's only
        *conditionally* required depending on a sibling value — without
        baking that domain knowledge into the schema validator itself. A
        failure here escalates to the next model in the chain exactly like a
        generic schema-validation failure (Groq only; Gemini's responseSchema
        is trusted as-is, unchanged).
        """
        if self.cfg.provider == "gemini":
            return self._json_call_gemini(system, user, schema, name, max_tokens)
        return self._json_call_groq(system, user, schema, name, max_tokens, extra_validate)

    def vision(self, prompt: str, image_path: str) -> str:
        if self.cfg.vision_provider == "gemini":
            return self._vision_gemini(prompt, image_path)
        return self._vision_groq(prompt, image_path)

    # ============================================================== groq ===
    def _chain(self) -> list[str]:
        seen: set[str] = set()
        chain = []
        for m in (self.cfg.model, self.cfg.fallback_model, self.cfg.strong_fallback_model):
            if m and m not in seen:
                seen.add(m)
                chain.append(m)
        if not chain:
            raise LLMRequestError("no Groq model configured (LLM_MODEL / LLM_FALLBACK_MODEL / "
                                   "LLM_STRONG_FALLBACK_MODEL are all empty)")
        return chain

    def _json_call_groq(self, system: str, user: str, schema: dict, name: str,
                         max_tokens: int | None = None, extra_validate=None) -> dict:
        last_err: Exception | None = None
        for model in self._chain():
            try:
                return self._cached(
                    {"system": system, "user": user, "schema": schema, "name": name, "model": model},
                    lambda m=model: self._call_groq_model(m, system, user, schema, name, max_tokens,
                                                           extra_validate),
                )
            except (LLMValidationError, LLMTransientError) as e:
                last_err = e
                if self.cfg.verbose:
                    print(f"LLM_FALLBACK from={model} reason={type(e).__name__} name={name}: {e}")
                continue
        raise LLMError(f"all configured Groq models failed for '{name}': {last_err}") from last_err

    def _call_groq_model(self, model: str, system: str, user: str, schema: dict, name: str,
                          max_tokens: int | None = None, extra_validate=None) -> dict:
        self.calls += 1
        sys_prompt = system + _schema_instructions(schema, name)
        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]

        if model in self.STRUCTURED_OUTPUT_MODELS:
            try:
                text = self._chat_groq(model, messages, {
                    "type": "json_schema",
                    "json_schema": {"name": name, "schema": _to_strict_openai_schema(schema), "strict": True},
                }, max_tokens=max_tokens)
            except LLMRequestError as e:
                # This model/deployment didn't accept our schema in strict
                # mode — same model, just fall back to plain JSON mode
                # rather than treating it as a hard config error. Logged
                # (not silent) so a live run can confirm whether a given
                # call actually got strict Structured Outputs or degraded.
                if self.cfg.verbose:
                    print(f"LLM_STRUCTURED_OUTPUT_DEGRADE model={model} name={name} reason={e}")
                text = self._chat_groq(model, messages, {"type": "json_object"}, max_tokens=max_tokens)
        else:
            text = self._chat_groq(model, messages, {"type": "json_object"}, max_tokens=max_tokens)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMValidationError(f"{model} returned malformed JSON: {e}") from None
        if not isinstance(parsed, dict):
            raise LLMValidationError(
                f"{model} returned a JSON {type(parsed).__name__}, expected an object")

        errors = _validate_json_schema(parsed, schema)
        if not errors and extra_validate is not None:
            errors = list(extra_validate(parsed))
        if errors:
            raise LLMValidationError(f"{model} response failed schema validation: {errors[:5]}")

        if self.cfg.verbose:
            print(f"LLM_OK provider=groq model={model} name={name}")
        return parsed

    def _chat_groq(self, model: str, messages: list, response_format: dict | None,
                   api_key: str | None = None, base_url: str | None = None,
                   max_tokens: int | None = None) -> str:
        api_key = api_key if api_key is not None else self.cfg.api_key
        base_url = base_url if base_url is not None else self.cfg.base_url
        max_tokens = max_tokens if max_tokens is not None else self.cfg.max_tokens
        url = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
        body_bytes = json.dumps(payload).encode("utf-8")

        delay = 2.0
        for attempt in range(self.cfg.max_retries):
            req = urllib.request.Request(
                url,
                data=body_bytes,
                headers={
                    "content-type": "application/json",
                    "authorization": f"Bearer {api_key}",
                    "user-agent": "agentic-bank/1.0",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return self._extract_groq_text(json.loads(r.read()), model)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                if e.code == 403 and _is_cloudflare_block(body):
                    raise LLMProviderBlockedError(
                        f"Groq request for model {model} was blocked by Cloudflare "
                        f"(HTTP 403, error code 1010) before reaching the model — this is "
                        f"provider/edge request-blocking, not an invalid API key. Common cause: "
                        f"a missing or blocked User-Agent header. {_safe_excerpt(body)}") from None
                if e.code in (401, 403):
                    raise LLMAuthError(
                        f"Groq auth error {e.code} for model {model}: {_safe_excerpt(body)}") from None
                if e.code == 429:
                    if attempt < self.cfg.max_retries - 1:
                        wait = _parse_retry_after(e.headers)
                        if wait is None:
                            wait = delay + random.uniform(0, 0.5)
                            delay = min(delay * 2, 30.0)
                        if self.cfg.verbose:
                            print(f"LLM_RETRY model={model} status=429 wait={wait:.1f}s "
                                  f"attempt={attempt + 1}/{self.cfg.max_retries}")
                        time.sleep(wait)
                        continue
                    raise LLMTransientError(
                        f"Groq rate limit exhausted for model {model} after "
                        f"{self.cfg.max_retries} attempts: {_safe_excerpt(body)}") from None
                if e.code == 413:
                    # HTTP 413 (rate_limit_exceeded / request too large) is Groq
                    # saying the request's tokens (prompt + requested max_tokens)
                    # exceed the model's per-minute budget outright — not a
                    # transient condition a retry/backoff will fix. Since
                    # max_tokens is already a small, task-sized budget by the
                    # time it reaches here, this means the PROMPT itself is the
                    # problem — surface that distinctly instead of retrying.
                    requested, limit = _parse_rate_limit_tokens(body)
                    detail = (f"requested={requested} limit={limit} max_tokens_asked={max_tokens} "
                              if requested is not None else f"max_tokens_asked={max_tokens} ")
                    raise LLMPromptTooLargeError(
                        f"Groq rejected model {model}'s request as too large for its per-minute "
                        f"token budget ({detail}HTTP 413). The completion budget requested here is "
                        f"already minimal, so the prompt/input itself is what needs to shrink (e.g. "
                        f"a smaller batch or document slice) — retrying will not help. "
                        f"{_safe_excerpt(body)}") from None
                if e.code == 400 and _parse_groq_error_code(body) == "json_validate_failed":
                    # Groq's JSON Object Mode itself failing to produce
                    # syntactically valid JSON (documented Groq behavior) is
                    # a model-output/generation-quality problem, not a
                    # malformed request from us — fallback-eligible, same as
                    # any other LLMValidationError. `failed_generation` in
                    # the body may contain document/banking content, so only
                    # a short excerpt is logged, and it's never parsed or
                    # treated as application data.
                    raise LLMValidationError(
                        f"{model} failed to generate valid JSON (Groq HTTP 400, "
                        f"error.code=json_validate_failed): {_safe_excerpt(body, 200)}") from None
                if e.code in (500, 502, 503, 504):
                    if attempt < self.cfg.max_retries - 1:
                        time.sleep(delay + random.uniform(0, 0.5))
                        delay = min(delay * 2, 30.0)
                        continue
                    raise LLMTransientError(
                        f"Groq server error {e.code} for model {model}: {_safe_excerpt(body)}") from None
                # Any other 400 (malformed payload, unsupported parameter,
                # invalid schema, bad model id) / 404 / 422 etc.: the request
                # itself is broken — a config/programming bug, never a
                # reason to retry or fall back.
                raise LLMRequestError(
                    f"Groq API error {e.code} for model {model}: {_safe_excerpt(body)}") from None
            except urllib.error.URLError:
                if attempt < self.cfg.max_retries - 1:
                    time.sleep(delay + random.uniform(0, 0.5))
                    delay = min(delay * 2, 30.0)
                    continue
                raise LLMTransientError(f"Groq API request failed: network error (model {model})") from None
        raise LLMTransientError(f"exhausted retries for model {model}")

    @staticmethod
    def _extract_groq_text(data: dict, model: str) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise LLMValidationError(f"{model} returned no choices")
        msg = choices[0].get("message") or {}
        text = msg.get("content")
        if not text or not str(text).strip():
            raise LLMValidationError(
                f"{model} returned empty output (finish_reason={choices[0].get('finish_reason')!r})")
        return text

    def _vision_groq(self, prompt: str, image_path: str) -> str:
        media = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        b64 = base64.b64encode(open(image_path, "rb").read()).decode()
        bypass = os.environ.get("AB_BYPASS_VISION_CACHE") == "1"

        def run():
            print(f"VISION_API_CALL file={image_path}")
            self.calls += 1
            self.vision_calls += 1
            messages = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}},
            ]}]
            return self._chat_groq(self.cfg.vision_model, messages, response_format=None,
                                    api_key=self.cfg.vision_api_key, base_url=self.cfg.vision_base_url,
                                    max_tokens=self.cfg.vision_max_tokens)

        def on_hit():
            print(f"VISION_CACHE_HIT file={image_path}")

        return self._cached(
            {"provider": self.cfg.vision_provider, "model": self.cfg.vision_model,
             "base_url": self.cfg.vision_base_url, "prompt": prompt,
             "image": hashlib.md5(b64.encode()).hexdigest()},
            run, bypass_read=bypass, on_hit=on_hit)

    # ============================================================ gemini ===
    def _post_gemini(self, payload: dict, model: str | None = None,
                      api_key: str | None = None, base_url: str | None = None) -> dict:
        model = model if model is not None else self.cfg.model
        api_key = api_key if api_key is not None else self.cfg.api_key
        base_url = base_url if base_url is not None else self.cfg.base_url
        url = f"{base_url}/v1beta/models/{model}:generateContent"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-goog-api-key": api_key,
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

    def _json_call_gemini(self, system: str, user: str, schema: dict, name: str,
                           max_tokens: int | None = None) -> dict:
        """Return a dict validated against `schema` by Gemini's structured output."""
        def run():
            self.calls += 1
            payload = {
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "systemInstruction": {"parts": [{"text": system}]},
                "generationConfig": {
                    "temperature": self.cfg.temperature,
                    "maxOutputTokens": max_tokens if max_tokens is not None else self.cfg.max_tokens,
                    "responseMimeType": "application/json",
                    "responseSchema": _to_gemini_schema(schema),
                },
            }
            data = self._post_gemini(payload)
            text = self._extract_text(data)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                raise LLMError(f"Gemini returned malformed JSON: {e}") from None
            if not isinstance(parsed, dict):
                raise LLMError(f"Gemini returned a JSON {type(parsed).__name__}, expected an object")
            return parsed

        return self._cached({"system": system, "user": user, "schema": schema, "name": name}, run)

    def _vision_gemini(self, prompt: str, image_path: str) -> str:
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
                    "maxOutputTokens": self.cfg.vision_max_tokens,
                },
            }
            data = self._post_gemini(payload, model=self.cfg.vision_model,
                                      api_key=self.cfg.vision_api_key, base_url=self.cfg.vision_base_url)
            return self._extract_text(data)

        def on_hit():
            print(f"VISION_CACHE_HIT file={image_path}")

        return self._cached(
            {"provider": self.cfg.vision_provider, "model": self.cfg.vision_model,
             "base_url": self.cfg.vision_base_url, "prompt": prompt,
             "image": hashlib.md5(b64.encode()).hexdigest()},
            run, bypass_read=bypass, on_hit=on_hit)


def _legacy_cache_key(cfg: LLMConfig, key: dict) -> dict | None:
    """Reconstruct the pre-fix cache key shape for one call, if it matches
    a known shape, so already-cached entries stay reachable. Returns None for
    key shapes that didn't exist under the old format (e.g. the transaction
    classifier's wire format changed, so there is nothing valid to recover)."""
    if {"system", "user", "schema", "name"} <= key.keys() and "model" not in key:
        return {"s": key["system"], "u": key["user"], "sc": key["schema"],
                "provider": cfg.provider, "m": cfg.model}
    if {"prompt", "image"} <= key.keys():
        return {"p": key["prompt"], "img": key["image"],
                "provider": cfg.provider, "m": cfg.model}
    return None


def _parse_groq_error_code(body: str) -> str | None:
    """The structured `error.code` out of a Groq JSON error body (e.g.
    '"error": {"code": "json_validate_failed", ...}'), or None if the body
    isn't JSON or has no such field. Only the code is read — the rest of
    the error body (which may embed `failed_generation`, i.e. real document
    content the model was working from) is never parsed as application
    data, only ever logged as a short excerpt."""
    try:
        return (json.loads(body).get("error") or {}).get("code")
    except (json.JSONDecodeError, AttributeError):
        return None


def _parse_rate_limit_tokens(body: str) -> tuple[int | None, int | None]:
    """Best-effort (requested, limit) token counts out of a Groq 413/429
    rate-limit error body, e.g. '...Requested 8121... Limit 6000...'.
    Returns (None, None) if the body doesn't have that shape (still safe to
    report the raw excerpt in that case)."""
    import re
    req_m = re.search(r"[Rr]equested\D{0,10}(\d+)", body)
    lim_m = re.search(r"[Ll]imit\D{0,10}(\d+)", body)
    return (int(req_m.group(1)) if req_m else None,
            int(lim_m.group(1)) if lim_m else None)


def _is_cloudflare_block(body: str) -> bool:
    """True if a 403 body looks like a Cloudflare edge block (error code
    1010, typically triggered by a missing/blocklisted User-Agent) rather
    than Groq itself rejecting the credentials."""
    low = body.lower()
    return "cloudflare" in low and "1010" in low


def _is_daily_quota_error(body: str) -> bool:
    low = body.lower().replace(" ", "")
    return "resource_exhausted" in low and "perday" in low


def _parse_retry_after(headers) -> float | None:
    """Seconds to wait, from a 429 response's `Retry-After` header, if present
    and parseable. Returns None to let the caller use its own backoff."""
    val = headers.get("Retry-After") if headers else None
    if val is None:
        return None
    try:
        return max(0.0, float(val))
    except ValueError:
        return None


def _safe_excerpt(body: str, limit: int = 400) -> str:
    """First `limit` chars of an error body, with anything key-shaped scrubbed."""
    import re
    scrubbed = re.sub(r'"?key"?\s*[:=]\s*"?[A-Za-z0-9_\-]{16,}"?', '"key": "<redacted>"', body,
                       flags=re.IGNORECASE)
    scrubbed = re.sub(r'(?i)authorization"?\s*[:=]\s*"?bearer\s+[A-Za-z0-9_\-.]+"?',
                       'authorization": "<redacted>"', scrubbed)
    return scrubbed[:limit]
