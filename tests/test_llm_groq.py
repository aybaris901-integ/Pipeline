"""Mocked tests for the Groq provider path in agentic_bank.llm / config.

No network access and no API key required — every HTTP call is faked via
`unittest.mock.patch` on `urllib.request.urlopen`. These must never consume
live API quota.

Run with:  python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations
import email.message
import importlib
import io
import json
import os
import unittest
import urllib.error
from unittest import mock

from agentic_bank import config as config_module
from agentic_bank.config import LLMConfig
from agentic_bank import llm as llm_module
from agentic_bank.llm import (
    LLM, LLMAuthError, LLMRequestError, LLMTransientError, LLMValidationError,
    LLMProviderBlockedError, LLMPromptTooLargeError,
)


SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"foo": {"type": "string"}},
    "required": ["foo"],
}


def _cfg(tmp_dir_unused=None, **overrides) -> LLMConfig:
    defaults = dict(
        provider="groq", model="llama-3.1-8b-instant",
        fallback_model="openai/gpt-oss-20b", strong_fallback_model="openai/gpt-oss-120b",
        api_key="test-key", base_url="https://groq.test/openai/v1",
        vision_api_key="test-key", vision_base_url="https://groq.test/openai/v1",
        max_retries=3, verbose=False,
    )
    defaults.update(overrides)
    return LLMConfig(**defaults)


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.body


def chat_ok(content: str) -> FakeResponse:
    body = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}).encode()
    return FakeResponse(body)


def http_error(code: int, body: dict | str = "error", retry_after: str | None = None) -> urllib.error.HTTPError:
    hdrs = email.message.Message()
    if retry_after is not None:
        hdrs["Retry-After"] = retry_after
    payload = body if isinstance(body, str) else json.dumps(body)
    return urllib.error.HTTPError(
        url="https://groq.test/openai/v1/chat/completions", code=code, msg="err",
        hdrs=hdrs, fp=io.BytesIO(payload.encode()),
    )


class ScriptedUrlopen:
    """Each call to urlopen() pops and executes the next scripted step."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []  # list of (url, parsed_request_body)

    def __call__(self, req, timeout=None):
        self.calls.append((req.full_url, json.loads(req.data), dict(req.headers)))
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def patched(steps):
    scripted = ScriptedUrlopen(steps)
    return mock.patch.object(llm_module.urllib.request, "urlopen", scripted), scripted


class TestConfig(unittest.TestCase):
    def _reload_with_env(self, env: dict) -> LLMConfig:
        with mock.patch.dict(os.environ, env, clear=True):
            importlib.reload(config_module)
            return config_module.LLMConfig()

    def tearDown(self):
        importlib.reload(config_module)  # restore real-environment defaults for later tests

    def test_defaults_are_groq_with_required_model_chain(self):
        cfg = self._reload_with_env({})
        self.assertEqual(cfg.provider, "groq")
        self.assertEqual(cfg.model, "llama-3.1-8b-instant")
        self.assertEqual(cfg.fallback_model, "openai/gpt-oss-20b")
        self.assertEqual(cfg.strong_fallback_model, "openai/gpt-oss-120b")
        self.assertEqual(cfg.base_url, "https://api.groq.com/openai/v1")
        self.assertEqual(cfg.vision_provider, "groq")
        # meta-llama/llama-4-scout-17b-16e-instruct was retired by Groq on
        # 2026-07-17 (deprecated/shut down) — qwen/qwen3.6-27b replaces it
        # as the vision/OCR default. The text chain above must NOT change.
        self.assertEqual(cfg.vision_model, "qwen/qwen3.6-27b")

    def test_env_vars_override_model_chain(self):
        cfg = self._reload_with_env({
            "GROQ_API_KEY": "k",
            "LLM_MODEL": "custom-primary",
            "LLM_FALLBACK_MODEL": "custom-fallback",
            "LLM_STRONG_FALLBACK_MODEL": "custom-strong",
        })
        self.assertEqual(cfg.model, "custom-primary")
        self.assertEqual(cfg.fallback_model, "custom-fallback")
        self.assertEqual(cfg.strong_fallback_model, "custom-strong")
        self.assertEqual(cfg.api_key, "k")

    def test_env_var_overrides_vision_model(self):
        cfg = self._reload_with_env({
            "GROQ_API_KEY": "k",
            "LLM_VISION_MODEL": "custom-vision-model",
        })
        self.assertEqual(cfg.vision_model, "custom-vision-model")
        self.assertEqual(cfg.model, "llama-3.1-8b-instant")  # text chain untouched

    def test_gemini_provider_still_supported(self):
        cfg = self._reload_with_env({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "gk"})
        self.assertEqual(cfg.provider, "gemini")
        self.assertEqual(cfg.api_key, "gk")
        self.assertEqual(cfg.base_url, "https://generativelanguage.googleapis.com")
        self.assertEqual(cfg.vision_provider, "gemini")  # follows provider when unset

    def test_unknown_provider_raises_clear_error(self):
        with self.assertRaises(ValueError):
            self._reload_with_env({"LLM_PROVIDER": "not-a-real-provider"})

    def test_vision_provider_follows_explicit_provider_override(self):
        # Not just the env var: an explicit provider= kwarg must be honoured too.
        cfg = LLMConfig(provider="gemini", api_key="x", base_url="https://g")
        self.assertEqual(cfg.vision_provider, "gemini")


class TestModelSelection(unittest.TestCase):
    def test_chain_dedupes_and_preserves_order(self):
        cfg = _cfg(model="a", fallback_model="a", strong_fallback_model="b")
        client = LLM(cfg, cache_dir=".cache_test_dedup")
        self.assertEqual(client._chain(), ["a", "b"])

    def test_empty_chain_is_a_clear_error(self):
        cfg = _cfg()
        # __post_init__ fills in a provider default when model="" is passed
        # in, so blank all three out afterwards to simulate a fully-empty
        # chain (e.g. every env var explicitly set to "").
        cfg.model = cfg.fallback_model = cfg.strong_fallback_model = ""
        client = LLM(cfg, cache_dir=".cache_test_empty")
        with self.assertRaises(LLMRequestError):
            client._chain()


class TestJsonCallGroq(unittest.TestCase):
    def setUp(self):
        self.cache_dir = os.path.join("tests", "_cache_tmp", self.id())
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_primary_model_success_no_fallback(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([chat_ok(json.dumps({"foo": "bar"}))])
        with patch_ctx:
            out = client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(out, {"foo": "bar"})
        self.assertEqual(len(scripted.calls), 1)
        self.assertEqual(scripted.calls[0][1]["model"], "llama-3.1-8b-instant")
        self.assertEqual(client.calls, 1)

    def test_malformed_json_escalates_to_fallback(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([
            chat_ok("not json at all"),                       # primary: bad JSON
            chat_ok(json.dumps({"foo": "bar"})),               # fallback: valid
        ])
        with patch_ctx:
            out = client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(out, {"foo": "bar"})
        models_tried = [c[1]["model"] for c in scripted.calls]
        self.assertEqual(models_tried, ["llama-3.1-8b-instant", "openai/gpt-oss-20b"])

    def test_schema_validation_failure_escalates(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([
            chat_ok(json.dumps({"wrong_field": 1})),          # primary: missing required 'foo'
            chat_ok(json.dumps({"foo": "bar"})),                # fallback: valid
        ])
        with patch_ctx:
            out = client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(out, {"foo": "bar"})
        self.assertEqual(len(scripted.calls), 2)

    def test_all_models_failing_raises(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([
            chat_ok("bad"), chat_ok("bad"), chat_ok("bad"),
        ])
        with patch_ctx:
            with self.assertRaises(llm_module.LLMError):
                client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(len(scripted.calls), 3)

    def test_auth_error_never_falls_back(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([http_error(401, "invalid api key")])
        with patch_ctx:
            with self.assertRaises(LLMAuthError):
                client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(len(scripted.calls), 1)  # never retried, never escalated

    def test_bad_request_never_falls_back(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([http_error(400, "model does not exist")])
        with patch_ctx:
            with self.assertRaises(LLMRequestError):
                client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(len(scripted.calls), 1)

    def test_429_honors_retry_after_then_succeeds(self):
        cfg = _cfg(max_retries=3)
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([
            http_error(429, "rate limited", retry_after="7"),
            chat_ok(json.dumps({"foo": "bar"})),
        ])
        with patch_ctx, mock.patch.object(llm_module.time, "sleep") as sleep_mock:
            out = client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(out, {"foo": "bar"})
        sleep_mock.assert_called_once_with(7.0)

    def test_429_exhausted_escalates_to_fallback(self):
        cfg = _cfg(max_retries=2)
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([
            http_error(429, "rate limited"),
            http_error(429, "rate limited"),
            chat_ok(json.dumps({"foo": "bar"})),
        ])
        with patch_ctx, mock.patch.object(llm_module.time, "sleep"):
            out = client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(out, {"foo": "bar"})
        models_tried = [c[1]["model"] for c in scripted.calls]
        self.assertEqual(models_tried, ["llama-3.1-8b-instant", "llama-3.1-8b-instant", "openai/gpt-oss-20b"])

    def test_gpt_oss_uses_structured_output_then_degrades_on_400(self):
        cfg = _cfg(model="openai/gpt-oss-20b", fallback_model="openai/gpt-oss-120b", strong_fallback_model="")
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([
            http_error(400, "schema not supported"),          # strict json_schema attempt rejected
            chat_ok(json.dumps({"foo": "bar"})),                # same model, json_object mode
        ])
        with patch_ctx:
            out = client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(out, {"foo": "bar"})
        self.assertEqual(len(scripted.calls), 2)
        first_fmt = scripted.calls[0][1]["response_format"]
        second_fmt = scripted.calls[1][1]["response_format"]
        self.assertEqual(first_fmt["type"], "json_schema")
        self.assertEqual(second_fmt["type"], "json_object")

    def test_cache_hit_makes_no_api_call(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([chat_ok(json.dumps({"foo": "bar"}))])
        with patch_ctx:
            client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(len(scripted.calls), 1)

        # Second call, same inputs: urlopen must not be invoked at all.
        with mock.patch.object(llm_module.urllib.request, "urlopen",
                                side_effect=AssertionError("must not call the network on a cache hit")):
            out = client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(out, {"foo": "bar"})

    def test_cache_key_distinguishes_model(self):
        cfg_a = _cfg(model="model-a", fallback_model="", strong_fallback_model="")
        cfg_b = _cfg(model="model-b", fallback_model="", strong_fallback_model="")
        client_a = LLM(cfg_a, self.cache_dir)
        client_b = LLM(cfg_b, self.cache_dir)
        key = {"system": "s", "user": "u", "schema": SIMPLE_SCHEMA, "name": "t", "model": "model-a"}
        key_b = {**key, "model": "model-b"}
        self.assertNotEqual(
            client_a._hash({"provider": "groq", "model": "model-a", "base_url": cfg_a.base_url, **key}),
            client_b._hash({"provider": "groq", "model": "model-b", "base_url": cfg_b.base_url, **key_b}),
        )


class TestVisionGroq(unittest.TestCase):
    def setUp(self):
        self.cache_dir = os.path.join("tests", "_cache_tmp", self.id())
        os.makedirs(self.cache_dir, exist_ok=True)
        self.png_path = os.path.join(self.cache_dir, "page.png")
        with open(self.png_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nfakepixels")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_vision_sends_image_url_payload_and_counts_real_call(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([chat_ok("transcribed text")])
        with patch_ctx:
            text = client.vision("transcribe this", self.png_path)
        self.assertEqual(text, "transcribed text")
        self.assertEqual(client.vision_calls, 1)
        content = scripted.calls[0][1]["messages"][0]["content"]
        kinds = {c["type"] for c in content}
        self.assertEqual(kinds, {"text", "image_url"})

    def test_vision_cache_hit_does_not_increment_vision_calls(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, _ = patched([chat_ok("transcribed text")])
        with patch_ctx:
            client.vision("transcribe this", self.png_path)
        self.assertEqual(client.vision_calls, 1)

        with mock.patch.object(llm_module.urllib.request, "urlopen",
                                side_effect=AssertionError("must not call the network on a cache hit")):
            text = client.vision("transcribe this", self.png_path)
        self.assertEqual(text, "transcribed text")
        self.assertEqual(client.vision_calls, 1)  # unchanged: no real call made


class TestSchemaValidation(unittest.TestCase):
    def test_valid_object_passes(self):
        self.assertEqual(llm_module._validate_json_schema({"foo": "x"}, SIMPLE_SCHEMA), [])

    def test_missing_required_fails(self):
        errors = llm_module._validate_json_schema({}, SIMPLE_SCHEMA)
        self.assertTrue(any("foo" in e for e in errors))

    def test_wrong_type_fails(self):
        errors = llm_module._validate_json_schema({"foo": 5}, SIMPLE_SCHEMA)
        self.assertTrue(errors)

    def test_nested_array_items_validated(self):
        schema = {"type": "object", "properties": {"items": {
            "type": "array", "items": {"type": "object",
                                        "properties": {"id": {"type": "string"}},
                                        "required": ["id"]}}}}
        errors = llm_module._validate_json_schema({"items": [{"id": "a"}, {}]}, schema)
        self.assertTrue(any("[1]" in e for e in errors))

    def test_null_for_optional_property_is_valid(self):
        # Mirrors _to_strict_openai_schema's own documented contract: an
        # optional property strict-mode models can't literally omit is
        # represented as null instead — the validator must accept that as
        # the omission it stands for, not reject it as a type mismatch.
        schema = {"type": "object", "properties": {
            "foo": {"type": "string"}, "bar": {"type": "object", "properties": {}}}}
        self.assertEqual(llm_module._validate_json_schema({"foo": None, "bar": None}, schema), [])

    def test_null_for_required_property_still_fails(self):
        # The omission-via-null contract only applies to OPTIONAL properties.
        # A required property set to null is still a genuine validation
        # failure, not something this change should start tolerating.
        errors = llm_module._validate_json_schema({"foo": None}, SIMPLE_SCHEMA)
        self.assertTrue(errors)

    def test_missing_required_property_still_reported_even_if_optional_nulls_are_fine(self):
        schema = {"type": "object", "properties": {
            "foo": {"type": "string"}, "bar": {"type": "string"}}, "required": ["foo"]}
        errors = llm_module._validate_json_schema({"bar": None}, schema)
        self.assertTrue(any("foo" in e for e in errors))


class TestUserAgentHeader(unittest.TestCase):
    """Groq's Cloudflare edge has been observed to 403 (error code 1010)
    requests sent with Python's default urllib User-Agent. Every Groq HTTP
    request must therefore carry an explicit User-Agent."""

    def setUp(self):
        self.cache_dir = os.path.join("tests", "_cache_tmp", self.id())
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_primary_model_request_sends_user_agent(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([chat_ok(json.dumps({"foo": "bar"}))])
        with patch_ctx:
            client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(scripted.calls[0][2].get("User-agent"), "agentic-bank/1.0")

    def test_fallback_and_strong_fallback_requests_send_user_agent(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([
            chat_ok("bad"), chat_ok("bad"), chat_ok(json.dumps({"foo": "bar"})),
        ])
        with patch_ctx:
            client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(len(scripted.calls), 3)  # primary + fallback + strong fallback
        for call in scripted.calls:
            self.assertEqual(call[2].get("User-agent"), "agentic-bank/1.0")

    def test_vision_request_sends_user_agent(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        png_path = os.path.join(self.cache_dir, "page.png")
        with open(png_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nfakepixels")
        patch_ctx, scripted = patched([chat_ok("transcribed text")])
        with patch_ctx:
            client.vision("transcribe this", png_path)
        self.assertEqual(scripted.calls[0][2].get("User-agent"), "agentic-bank/1.0")

    def test_preserves_authorization_and_content_type_headers(self):
        cfg = _cfg(api_key="secret-key")
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([chat_ok(json.dumps({"foo": "bar"}))])
        with patch_ctx:
            client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        headers = scripted.calls[0][2]
        self.assertEqual(headers.get("Authorization"), "Bearer secret-key")
        self.assertEqual(headers.get("Content-type"), "application/json")


class TestCloudflareBlockClassification(unittest.TestCase):
    """A Cloudflare edge block (403 + error code 1010) is a distinct failure
    mode from an invalid API key and must be reported as such."""

    def setUp(self):
        self.cache_dir = os.path.join("tests", "_cache_tmp", self.id())
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_cloudflare_1010_is_not_reported_as_invalid_api_key(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        body = (
            "<html><head><title>Access denied | groq.test used Cloudflare "
            "to restrict access</title></head><body>Error code: 1010"
            "<div>cloudflare</div></body></html>"
        )
        patch_ctx, scripted = patched([http_error(403, body)])
        with patch_ctx:
            with self.assertRaises(LLMProviderBlockedError):
                client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(len(scripted.calls), 1)  # never retried, never escalated

    def test_plain_403_without_cloudflare_marker_is_still_auth_error(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([http_error(403, "invalid api key")])
        with patch_ctx:
            with self.assertRaises(LLMAuthError):
                client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(len(scripted.calls), 1)


class TestMaxTokensBudget(unittest.TestCase):
    """A tiny request must not reserve thousands of output tokens — that's
    what tripped Groq's free-tier TPM limit (HTTP 413) even though the
    prompt itself was small."""

    def setUp(self):
        self.cache_dir = os.path.join("tests", "_cache_tmp", self.id())
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_default_max_tokens_is_small_not_8000(self):
        cfg = _cfg()
        self.assertLessEqual(cfg.max_tokens, 2048)
        self.assertGreater(cfg.max_tokens, 0)

    def test_json_call_without_override_uses_bounded_default(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([chat_ok(json.dumps({"foo": "bar"}))])
        with patch_ctx:
            client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        sent = scripted.calls[0][1]["max_tokens"]
        self.assertEqual(sent, cfg.max_tokens)
        self.assertLess(sent, 4000)  # nowhere near the old ~8000 default

    def test_json_call_honors_explicit_max_tokens_override(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([chat_ok(json.dumps({"foo": "bar"}))])
        with patch_ctx:
            client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t", max_tokens=1024)
        self.assertEqual(scripted.calls[0][1]["max_tokens"], 1024)

    def test_override_applies_to_every_model_in_the_fallback_chain(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([
            chat_ok("bad"), chat_ok("bad"), chat_ok(json.dumps({"foo": "bar"})),
        ])
        with patch_ctx:
            client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t", max_tokens=1024)
        self.assertEqual(len(scripted.calls), 3)
        for call in scripted.calls:
            self.assertEqual(call[1]["max_tokens"], 1024)

    def test_vision_uses_its_own_larger_budget_independent_of_json_calls(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        png_path = os.path.join(self.cache_dir, "page.png")
        with open(png_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\nfakepixels")
        patch_ctx, scripted = patched([chat_ok("transcribed text")])
        with patch_ctx:
            client.vision("transcribe this", png_path)
        self.assertEqual(scripted.calls[0][1]["max_tokens"], cfg.vision_max_tokens)
        self.assertNotEqual(cfg.vision_max_tokens, cfg.max_tokens)


class TestExtractCallSitesRequestBoundedTokens(unittest.TestCase):
    """extract.py's real call sites must each request a task-appropriate
    (not the old blanket 8000) completion budget."""

    def setUp(self):
        self.cache_dir = os.path.join("tests", "_cache_tmp", self.id())
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_kyc_and_adjustments_request_1024(self):
        from agentic_bank import extract
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([
            chat_ok(json.dumps({"ownership_threshold_pct": 20, "holdings": []})),
        ])
        with patch_ctx:
            extract.extract_kyc(client, "kyc text")
        self.assertEqual(scripted.calls[0][1]["max_tokens"], 1024)

        patch_ctx, scripted = patched([
            chat_ok(json.dumps({"is_binding": True, "adjustments": []})),
        ])
        with patch_ctx:
            extract.extract_adjustments(client, "adjustments text")
        self.assertEqual(scripted.calls[0][1]["max_tokens"], 1024)

    def test_covenants_and_classify_request_2048(self):
        from agentic_bank import extract
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([chat_ok(json.dumps({"covenants": []}))])
        with patch_ctx:
            extract.extract_covenants(client, "Статья 6 — Финансовые ковенанты\n...")
        self.assertEqual(scripted.calls[0][1]["max_tokens"], 2048)

        rows = [{"txn_id": "T1", "date": "2026-01-01", "amount": 1, "currency": "USD",
                 "counterparty": "x", "description": "y"}]
        patch_ctx, scripted = patched([
            chat_ok(json.dumps({"transactions": [{"id": "T1", "category": "other", "filler": False}]})),
        ])
        with patch_ctx:
            extract.classify_transactions(client, rows)
        self.assertEqual(scripted.calls[0][1]["max_tokens"], 2048)


class TestExtraValidateHook(unittest.TestCase):
    """json_call's optional extra_validate callback: a business-rule check
    (e.g. a field only conditionally required, which the generic JSON-Schema
    subset can't express) that must behave exactly like a generic
    schema-validation failure — escalate through the model chain, don't
    silently accept."""

    def setUp(self):
        self.cache_dir = os.path.join("tests", "_cache_tmp", self.id())
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_extra_validate_failure_escalates_like_schema_failure(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([
            chat_ok(json.dumps({"foo": "bad"})),
            chat_ok(json.dumps({"foo": "good"})),
        ])
        always_reject_bad = lambda parsed: (["foo must be 'good'"] if parsed.get("foo") != "good" else [])
        with patch_ctx:
            out = client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t", extra_validate=always_reject_bad)
        self.assertEqual(out, {"foo": "good"})
        self.assertEqual(len(scripted.calls), 2)

    def test_extra_validate_not_called_when_schema_validation_already_failed(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        calls = []

        def spy(parsed):
            calls.append(parsed)
            return []

        patch_ctx, scripted = patched([chat_ok(json.dumps({"wrong_field": 1}))])
        with patch_ctx:
            with self.assertRaises(LLMValidationError):
                client._call_groq_model("llama-3.1-8b-instant", "sys", "usr", SIMPLE_SCHEMA, "t",
                                         extra_validate=spy)
        self.assertEqual(calls, [])  # short-circuited: never reached extra_validate

    def test_extra_validate_passing_is_a_no_op(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([chat_ok(json.dumps({"foo": "bar"}))])
        with patch_ctx:
            out = client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t", extra_validate=lambda parsed: [])
        self.assertEqual(out, {"foo": "bar"})
        self.assertEqual(len(scripted.calls), 1)


class TestPromptTooLarge(unittest.TestCase):
    """A 413 (request too large for the model's per-minute token budget) is
    a distinct, non-retryable failure mode — not a generic request error and
    not something a retry/backoff loop can fix."""

    def setUp(self):
        self.cache_dir = os.path.join("tests", "_cache_tmp", self.id())
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_413_raises_prompt_too_large_without_retrying(self):
        cfg = _cfg(max_retries=3)
        client = LLM(cfg, self.cache_dir)
        body = json.dumps({"error": {
            "message": "Request too large for model `llama-3.1-8b-instant` in organization "
                       "... Limit 6000, Requested 8121.",
            "type": "tokens", "code": "rate_limit_exceeded",
        }})
        patch_ctx, scripted = patched([http_error(413, body)])
        with patch_ctx, mock.patch.object(llm_module.time, "sleep") as sleep_mock:
            with self.assertRaises(LLMPromptTooLargeError):
                client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(len(scripted.calls), 1)  # never retried
        sleep_mock.assert_not_called()

    def test_413_never_escalates_to_fallback_model(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        body = "Limit 6000, Requested 8121"
        patch_ctx, scripted = patched([http_error(413, body)])
        with patch_ctx:
            with self.assertRaises(LLMPromptTooLargeError):
                client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(len(scripted.calls), 1)  # fallback/strong-fallback never tried

    def test_413_message_reports_requested_and_limit(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        body = "blah Limit 6000, Requested 8121 blah"
        patch_ctx, scripted = patched([http_error(413, body)])
        with patch_ctx:
            with self.assertRaises(LLMPromptTooLargeError) as ctx:
                client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        msg = str(ctx.exception)
        self.assertIn("8121", msg)
        self.assertIn("6000", msg)


def _json_validate_failed_body(fake_generation: str = "not valid json {{{") -> str:
    return json.dumps({"error": {
        "message": "Failed to generate JSON. Please adjust your prompt.",
        "type": "invalid_request_error",
        "code": "json_validate_failed",
        "failed_generation": fake_generation,
    }})


class TestJsonValidateFailed(unittest.TestCase):
    """Groq's documented JSON Object Mode failure (HTTP 400,
    error.code=json_validate_failed): the model itself failed to produce
    syntactically valid JSON. That's a model-output/generation-quality
    problem — fallback-eligible, same as malformed JSON we catch ourselves
    after a 200 — never a fatal request error."""

    def setUp(self):
        self.cache_dir = os.path.join("tests", "_cache_tmp", self.id())
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_A_json_validate_failed_escalates_to_fallback_and_succeeds(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([
            http_error(400, _json_validate_failed_body()),          # llama: json_validate_failed
            chat_ok(json.dumps({"foo": "bar"})),                     # gpt-oss-20b: succeeds
        ])
        with patch_ctx:
            out = client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(out, {"foo": "bar"})
        models_tried = [c[1]["model"] for c in scripted.calls]
        self.assertEqual(models_tried, ["llama-3.1-8b-instant", "openai/gpt-oss-20b"])

    def test_A_all_three_models_json_validate_failed_raises_after_full_chain(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([
            http_error(400, _json_validate_failed_body()),
            http_error(400, _json_validate_failed_body()),
            http_error(400, _json_validate_failed_body()),
        ])
        with patch_ctx:
            with self.assertRaises(llm_module.LLMError):
                client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(len(scripted.calls), 3)  # every model in the chain was tried

    def test_A_failed_generation_content_is_not_logged_in_full(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        sensitive = "ACC-999999 secret borrower payment detail " * 20
        patch_ctx, scripted = patched([http_error(400, _json_validate_failed_body(fake_generation=sensitive))])
        with patch_ctx:
            with self.assertRaises(LLMValidationError) as ctx:
                client._call_groq_model("llama-3.1-8b-instant", "sys", "usr", SIMPLE_SCHEMA, "t")
        msg = str(ctx.exception)
        self.assertNotIn(sensitive, msg)
        self.assertLess(len(msg), len(sensitive))

    def test_B_ordinary_malformed_400_is_fatal_no_fallback(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        body = json.dumps({"error": {
            "message": "'foo' is not a valid parameter",
            "type": "invalid_request_error", "code": "invalid_value",
        }})
        patch_ctx, scripted = patched([http_error(400, body)])
        with patch_ctx:
            with self.assertRaises(LLMRequestError):
                client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(len(scripted.calls), 1)  # never retried, never escalated

    def test_B_400_with_unstructured_body_is_still_fatal(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([http_error(400, "plain text error, not JSON at all")])
        with patch_ctx:
            with self.assertRaises(LLMRequestError):
                client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(len(scripted.calls), 1)

    def test_B_400_with_different_error_code_is_still_fatal(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        body = json.dumps({"error": {"message": "model does not exist", "code": "model_not_found"}})
        patch_ctx, scripted = patched([http_error(400, body)])
        with patch_ctx:
            with self.assertRaises(LLMRequestError):
                client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(len(scripted.calls), 1)

    def test_C_401_still_never_falls_back(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([http_error(401, "invalid api key")])
        with patch_ctx:
            with self.assertRaises(LLMAuthError):
                client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(len(scripted.calls), 1)

    def test_C_403_still_never_falls_back(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        patch_ctx, scripted = patched([http_error(403, "forbidden")])
        with patch_ctx:
            with self.assertRaises(LLMAuthError):
                client.json_call("sys", "usr", SIMPLE_SCHEMA, name="t")
        self.assertEqual(len(scripted.calls), 1)


if __name__ == "__main__":
    unittest.main()
