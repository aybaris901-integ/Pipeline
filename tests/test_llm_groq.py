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
        self.calls.append((req.full_url, json.loads(req.data)))
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


if __name__ == "__main__":
    unittest.main()
