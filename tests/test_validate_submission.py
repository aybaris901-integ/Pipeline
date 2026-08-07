"""Regression tests for validate.py's `actual` contract.

The live crash ("'actual' must be greater than zero, got 0.0") happened
because evaluate.py's own metric() can legitimately produce a bare 0.0 for
a covenant leg that matches no transactions — and the OUTPUT contract (see
test_evaluate_actual.py) now folds that into None, the same "nothing
meaningful to report" representation metric() already uses when a ratio's
denominator is zero. This validator must accept that None, while still
rejecting zero, negative, and non-finite values as they are never a valid
*reported* measurement.

Uses a synthetic template/submission shape (not the real 12-scenario
dataset) so these tests prove the general contract, not one dataset's
specific cell count — EXPECTED_CELLS is patched to match the synthetic
shape used in each test.
"""
from __future__ import annotations
import unittest
from unittest import mock

from agentic_bank import validate as validate_module
from agentic_bank.validate import SubmissionError, validate_submission


def _template(clauses: list[str]) -> dict:
    return {"answers": {"S1": {c: {} for c in clauses}}}


def _submission(cells: dict) -> dict:
    return {"team": "t", "contact_email": "a@b.com", "model": "m",
            "answers": {"S1": cells}}


def _cell(status="COMPLIANT", actual=100.0, evidence=None) -> dict:
    return {"status": status, "actual": actual, "evidence_txn_id": evidence}


class TestActualAcceptsNull(unittest.TestCase):
    def test_null_actual_is_valid_no_meaningful_measurement(self):
        template = _template(["C1"])
        sub = _submission({"C1": _cell(actual=None)})
        with mock.patch.object(validate_module, "EXPECTED_CELLS", 1):
            validate_submission(sub, template)  # must not raise

    def test_genuine_positive_actual_is_valid(self):
        template = _template(["C1"])
        sub = _submission({"C1": _cell(actual=123.45)})
        with mock.patch.object(validate_module, "EXPECTED_CELLS", 1):
            validate_submission(sub, template)  # must not raise

    def test_zero_actual_is_still_rejected(self):
        template = _template(["C1"])
        sub = _submission({"C1": _cell(actual=0.0)})
        with mock.patch.object(validate_module, "EXPECTED_CELLS", 1):
            with self.assertRaises(SubmissionError):
                validate_submission(sub, template)

    def test_negative_actual_is_still_rejected(self):
        template = _template(["C1"])
        sub = _submission({"C1": _cell(actual=-5.0)})
        with mock.patch.object(validate_module, "EXPECTED_CELLS", 1):
            with self.assertRaises(SubmissionError):
                validate_submission(sub, template)

    def test_nan_actual_is_still_rejected(self):
        template = _template(["C1"])
        sub = _submission({"C1": _cell(actual=float("nan"))})
        with mock.patch.object(validate_module, "EXPECTED_CELLS", 1):
            with self.assertRaises(SubmissionError):
                validate_submission(sub, template)

    def test_infinite_actual_is_still_rejected(self):
        template = _template(["C1"])
        sub = _submission({"C1": _cell(actual=float("inf"))})
        with mock.patch.object(validate_module, "EXPECTED_CELLS", 1):
            with self.assertRaises(SubmissionError):
                validate_submission(sub, template)

    def test_bool_actual_is_still_rejected(self):
        template = _template(["C1"])
        sub = _submission({"C1": _cell(actual=True)})
        with mock.patch.object(validate_module, "EXPECTED_CELLS", 1):
            with self.assertRaises(SubmissionError):
                validate_submission(sub, template)

    def test_string_actual_is_still_rejected(self):
        template = _template(["C1"])
        sub = _submission({"C1": _cell(actual="1.0")})
        with mock.patch.object(validate_module, "EXPECTED_CELLS", 1):
            with self.assertRaises(SubmissionError):
                validate_submission(sub, template)

    def test_mixed_null_and_positive_cells_across_a_full_submission(self):
        template = _template(["C1", "C2", "C3"])
        sub = _submission({
            "C1": _cell(actual=None),
            "C2": _cell(actual=50.0),
            "C3": _cell(actual=None),
        })
        with mock.patch.object(validate_module, "EXPECTED_CELLS", 3):
            validate_submission(sub, template)  # must not raise


if __name__ == "__main__":
    unittest.main()
