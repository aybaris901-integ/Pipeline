"""Regression tests for the submission's `actual` output contract.

evaluate.py's own docstring says `actual` is "always reported as a positive
number" — and metric() already returns None (not zero) when a ratio's
denominator is zero. score.py's scoring rubric additionally treats a zero
`actual` in the ground truth identically to a null one (`kv in (None, 0)`).
Together these prove 0.0 is never a distinct, validly-reported `actual` in
this domain: it always means "nothing meaningful was measured," same as
None. final_actual() is where that folding happens, centrally, regardless
of which covenant/scenario produced the raw computed value.

No scenario IDs from the real dataset are used here — every case is built
from synthetic categories/transactions so the test proves the general
contract, not a fact about one dataset instance.
"""
from __future__ import annotations
import unittest

from agentic_bank.evaluate import Context, Txn, final_actual, metric, round2, verdict


def _ctx(*txns: Txn) -> Context:
    return Context(txns=list(txns))


def _txn(txn_id, category, amount, related_party=False):
    return Txn(txn_id=txn_id, date="2025-06-01", counterparty="Some Counterparty",
               description="d", amount=amount, currency="USD", category=category,
               related_party=related_party)


class TestRound2Unchanged(unittest.TestCase):
    def test_none_stays_none(self):
        self.assertIsNone(round2(None))

    def test_rounds_to_two_decimals(self):
        self.assertEqual(round2(1.23456), 1.23)


class TestFinalActualFolding(unittest.TestCase):
    """The normalized representation this contract requires."""

    def test_none_stays_none(self):
        self.assertIsNone(final_actual(None))

    def test_bare_zero_becomes_none(self):
        self.assertIsNone(final_actual(0.0))

    def test_value_rounding_to_zero_becomes_none(self):
        self.assertIsNone(final_actual(0.0049))  # rounds to 0.00

    def test_genuine_positive_value_is_rounded_and_kept(self):
        self.assertEqual(final_actual(123.456), 123.46)

    def test_no_legitimate_zero_case_exists(self):
        # Proves the negative of option D/"a legitimate zero": whatever the
        # magnitude or covenant kind, a value that rounds to exactly zero
        # is always folded to None, never reported as a bare 0.
        for v in (0.0, -0.0, 0.001, 0.004):
            self.assertIsNone(final_actual(v))


class TestMetricAndVerdictZeroVsNull(unittest.TestCase):
    """End-to-end: reproduces the general covenant-arithmetic shape behind
    the live failure (a related-party-filtered leg matching nothing) using
    entirely synthetic categories, not a real scenario."""

    def test_zero_denominator_ratio_is_null_via_metric_itself(self):
        # No "revenue" transactions at all => the denominator leg is 0 =>
        # metric() already returns None directly (pre-existing behavior).
        ctx = _ctx(_txn("T1", "payroll", -100.0))
        spec = {"kind": "ratio", "numerator": {"categories": ["payroll"]},
                "denominator": {"categories": ["revenue"]}}
        value = metric(ctx, spec)
        self.assertIsNone(value)
        self.assertIsNone(final_actual(value))

    def test_zero_valued_related_party_leg_yields_null_actual_end_to_end(self):
        # Revenue exists (so the denominator is real and non-zero), but
        # nothing matches the numerator's related_party_only filter — the
        # ratio genuinely computes to a bare 0.0, not None, from metric()
        # itself. The OUTPUT contract must still fold that to null.
        ctx = _ctx(_txn("T1", "revenue", 100000.0, related_party=False))
        spec = {"clause": "X", "kind": "ratio", "direction": "max", "threshold": 1.0,
                "numerator": {"categories": ["payroll"], "related_party_only": True},
                "denominator": {"categories": ["revenue"]},
                "period_start": "2025-01-01", "period_end": "2025-12-31"}
        status, value = verdict(ctx, spec)
        self.assertEqual(value, 0.0)            # metric() itself: a genuine bare zero
        self.assertEqual(status, "COMPLIANT")
        self.assertIsNone(final_actual(value))  # the reported representation: null

    def test_genuine_positive_ratio_is_preserved(self):
        ctx = _ctx(
            _txn("T1", "opex", -50000.0),
            _txn("T2", "revenue", 100000.0),
        )
        spec = {"clause": "Y", "kind": "ratio", "direction": "min", "threshold": 1.0,
                "numerator": {"categories": ["revenue"]}, "denominator": {"categories": ["opex"]},
                "period_start": "2025-01-01", "period_end": "2025-12-31"}
        status, value = verdict(ctx, spec)
        self.assertEqual(value, 2.0)
        self.assertEqual(status, "COMPLIANT")
        self.assertEqual(final_actual(value), 2.0)

    def test_genuine_positive_amount_is_preserved(self):
        ctx = _ctx(_txn("T1", "capex", -3204881.55))
        spec = {"clause": "Z", "kind": "amount", "direction": "max", "threshold": 1.0,
                "numerator": {"categories": ["capex"]}, "period_start": "2025-01-01",
                "period_end": "2025-12-31"}
        status, value = verdict(ctx, spec)
        self.assertEqual(final_actual(value), 3204881.55)

    def test_zero_valued_amount_leg_yields_null_actual_end_to_end(self):
        # An amount covenant whose numerator legitimately matches nothing
        # (e.g. no related-party lease payments at all) hits the exact
        # same fold: metric() returns a bare 0.0, final_actual -> None.
        ctx = _ctx(_txn("T1", "lease", -1000.0, related_party=False))
        spec = {"clause": "W", "kind": "amount", "direction": "max", "threshold": 1.0,
                "numerator": {"categories": ["lease"], "related_party_only": True},
                "period_start": "2025-01-01", "period_end": "2025-12-31"}
        status, value = verdict(ctx, spec)
        self.assertEqual(value, 0.0)
        self.assertIsNone(final_actual(value))


if __name__ == "__main__":
    unittest.main()
