"""Regression tests for extract.py's covenant null-value contract.

Reproduces the exact live failure (all three Groq models independently
emitting `denominator: null` / `notes: null`) and pins down the intended,
downstream-proven-safe resolution:

- denominator=null is VALID for kind="amount" — evaluate.metric() never
  reads denominator for an amount covenant, and covenant_rules.py (the
  offline/rules backend for the same domain) never even sets the key for
  its amount archetypes, i.e. omission is that backend's own canonical
  representation, and omitted-key vs. explicit-null are indistinguishable
  to dict.get().
- denominator=null is INVALID for kind="ratio" — evaluate.metric() divides
  by it there, so a missing denominator must fail loudly (escalate through
  the model chain, then error out) rather than silently evaluate to a
  vacuous "COMPLIANT".
- notes=null is valid unconditionally — no code anywhere reads a covenant's
  `notes` field.

No network access: every HTTP call is faked exactly like test_llm_groq.py.
"""
from __future__ import annotations
import json
import os
import unittest

from agentic_bank import extract
from agentic_bank.llm import LLM, LLMError
from tests.test_llm_groq import _cfg, chat_ok, patched


def _covenant(clause="6.1", kind="amount", threshold=1.0, denominator=None, notes=None):
    return {
        "clause": clause, "kind": kind, "direction": "max", "threshold": threshold,
        "numerator": {"categories": ["revenue"]},
        "denominator": denominator,
        "period_start": "2025-01-01", "period_end": "2025-12-31",
        "notes": notes,
    }


AGREEMENT_TEXT = "Статья 6 — Финансовые ковенанты\nПункт 6.1 текст пункта ..."


class TestCovenantNullContract(unittest.TestCase):
    def setUp(self):
        self.cache_dir = os.path.join("tests", "_cache_tmp", self.id())
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_amount_covenant_with_null_denominator_extracts_cleanly(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        body = json.dumps({"covenants": [_covenant(kind="amount", denominator=None)]})
        patch_ctx, scripted = patched([chat_ok(body)])
        with patch_ctx:
            covenants = extract.extract_covenants(client, AGREEMENT_TEXT)
        self.assertEqual(len(covenants), 1)
        self.assertIsNone(covenants[0]["denominator"])
        self.assertEqual(len(scripted.calls), 1)  # accepted on the primary model, no escalation

    def test_covenant_with_null_notes_extracts_cleanly(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        body = json.dumps({"covenants": [_covenant(
            kind="ratio", denominator={"categories": ["opex"]}, notes=None)]})
        patch_ctx, scripted = patched([chat_ok(body)])
        with patch_ctx:
            covenants = extract.extract_covenants(client, AGREEMENT_TEXT)
        self.assertIsNone(covenants[0]["notes"])
        self.assertEqual(len(scripted.calls), 1)

    def test_ratio_covenant_with_null_denominator_is_rejected_and_escalates(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        bad = json.dumps({"covenants": [_covenant(kind="ratio", denominator=None)]})
        # Reproduces the live failure: all three models in the chain
        # independently produce the same bad shape. Extraction must fail
        # loudly rather than silently accept a ratio with no denominator.
        patch_ctx, scripted = patched([chat_ok(bad), chat_ok(bad), chat_ok(bad)])
        with patch_ctx:
            with self.assertRaises(LLMError):
                extract.extract_covenants(client, AGREEMENT_TEXT)
        self.assertEqual(len(scripted.calls), 3)  # tried every model, never silently accepted

    def test_ratio_covenant_with_real_denominator_still_works(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        good = json.dumps({"covenants": [_covenant(
            kind="ratio", denominator={"categories": ["opex"]})]})
        patch_ctx, scripted = patched([chat_ok(good)])
        with patch_ctx:
            covenants = extract.extract_covenants(client, AGREEMENT_TEXT)
        self.assertEqual(covenants[0]["denominator"], {"categories": ["opex"]})
        self.assertEqual(len(scripted.calls), 1)

    def test_semantic_errors_helper_flags_only_ratio_with_null_denominator(self):
        parsed = {"covenants": [
            _covenant(kind="amount", denominator=None),
            _covenant(kind="ratio", denominator={"categories": ["opex"]}),
            _covenant(kind="ratio", denominator=None),
        ]}
        errors = extract._covenant_semantic_errors(parsed)
        self.assertEqual(len(errors), 1)
        self.assertIn("[2]", errors[0])


if __name__ == "__main__":
    unittest.main()
