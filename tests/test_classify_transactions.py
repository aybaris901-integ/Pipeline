"""Regression tests for extract.py's empty-amount ledger contract.

The master ledger ships a handful of rows with amount="" — not corrupt
data: the real figure is disclosed in a treasury memo or auditor note and
patched onto the transaction later via a `set_amount` adjustment in
run.build_context() (see the identical, already-established contract in
rules._amt / run._ledger_amount). classify_transactions() must therefore
treat amount="" as 0.0 for classification purposes (never as a reason to
crash, drop the row, or invent a non-zero value), exactly like the rules
backend already does for the same rows.

No network access: every HTTP call is faked exactly like test_llm_groq.py.
"""
from __future__ import annotations
import json
import os
import unittest

from agentic_bank import extract
from agentic_bank.llm import LLM
from tests.test_llm_groq import _cfg, chat_ok, patched


def _row(txn_id, amount, description="Payment", counterparty="Some LLP",
         date="2025-11-18", currency="USD"):
    return {"txn_id": txn_id, "date": date, "counterparty": counterparty,
            "description": description, "amount": amount, "currency": currency}


class TestAmtHelper(unittest.TestCase):
    """The normalized representation: same contract as rules._amt /
    run._ledger_amount — empty/unparseable amount -> 0.0, everything else
    parses as a plain float."""

    def test_empty_string_is_zero(self):
        self.assertEqual(extract._amt(""), 0.0)

    def test_whitespace_only_is_zero(self):
        self.assertEqual(extract._amt("   "), 0.0)

    def test_non_numeric_is_zero(self):
        self.assertEqual(extract._amt("n/a"), 0.0)

    def test_real_amount_parses(self):
        self.assertEqual(extract._amt("486204.19"), 486204.19)

    def test_negative_amount_parses(self):
        self.assertEqual(extract._amt("-500.00"), -500.0)

    def test_matches_rules_backend_contract(self):
        from agentic_bank import rules
        for v in ("", "   ", "not-a-number", "123.45", "-99.99"):
            self.assertEqual(extract._amt(v), rules._amt(v))


class TestClassifyTransactionsEmptyAmount(unittest.TestCase):
    """Reproduces the exact live crash: a real ledger row with amount=""
    reaching classify_transactions() before any LLM call is made."""

    def setUp(self):
        self.cache_dir = os.path.join("tests", "_cache_tmp", self.id())
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def test_empty_amount_row_does_not_raise_value_error(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        rows = [
            _row("TXN-P7-0032", "1000.00", description="Ordinary opex payment"),
            _row("TXN-P7-0033", "", description="Mineral extraction tax assessment 2025",
                 counterparty="State Revenue Committee"),
            _row("TXN-P7-0034", "-500.00", description="Another payment"),
        ]
        response = json.dumps({"transactions": [
            {"id": "TXN-P7-0032", "category": "opex", "filler": False},
            {"id": "TXN-P7-0033", "category": "tax", "filler": False},
            {"id": "TXN-P7-0034", "category": "opex", "filler": False},
        ]})
        patch_ctx, scripted = patched([chat_ok(response)])
        with patch_ctx:
            result = extract.classify_transactions(client, rows)  # must not raise ValueError
        self.assertEqual(len(scripted.calls), 1)
        self.assertEqual(result["TXN-P7-0033"]["category"], "tax")

    def test_empty_amount_formats_as_zero_in_the_prompt_not_dropped(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        rows = [
            _row("TXN-P7-0033", "", description="Mineral extraction tax assessment 2025",
                 counterparty="State Revenue Committee", date="2025-11-18"),
        ]
        response = json.dumps({"transactions": [
            {"id": "TXN-P7-0033", "category": "tax", "filler": False},
        ]})
        patch_ctx, scripted = patched([chat_ok(response)])
        with patch_ctx:
            extract.classify_transactions(client, rows)
        sent_lines = scripted.calls[0][1]["messages"][1]["content"]
        self.assertIn("TXN-P7-0033 | 2025-11-18 | 0.00 USD", sent_lines)

    def test_all_transaction_ids_preserved_none_silently_dropped(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        rows = [
            _row("TXN-A", "10.00"),
            _row("TXN-B", ""),
            _row("TXN-C", "30.00"),
        ]
        response = json.dumps({"transactions": [
            {"id": "TXN-A", "category": "opex", "filler": False},
            {"id": "TXN-B", "category": "tax", "filler": False},
            {"id": "TXN-C", "category": "opex", "filler": False},
        ]})
        patch_ctx, scripted = patched([chat_ok(response)])
        with patch_ctx:
            result = extract.classify_transactions(client, rows)
        self.assertEqual(set(result.keys()), {"TXN-A", "TXN-B", "TXN-C"})

    def test_row_order_preserved_in_the_prompt(self):
        cfg = _cfg()
        client = LLM(cfg, self.cache_dir)
        rows = [_row("TXN-A", "10.00"), _row("TXN-B", ""), _row("TXN-C", "30.00")]
        response = json.dumps({"transactions": [
            {"id": "TXN-A", "category": "opex", "filler": False},
            {"id": "TXN-B", "category": "tax", "filler": False},
            {"id": "TXN-C", "category": "opex", "filler": False},
        ]})
        patch_ctx, scripted = patched([chat_ok(response)])
        with patch_ctx:
            extract.classify_transactions(client, rows)
        sent_lines = scripted.calls[0][1]["messages"][1]["content"]
        self.assertLess(sent_lines.index("TXN-A"), sent_lines.index("TXN-B"))
        self.assertLess(sent_lines.index("TXN-B"), sent_lines.index("TXN-C"))


if __name__ == "__main__":
    unittest.main()
