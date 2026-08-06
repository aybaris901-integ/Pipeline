"""Stage 6 — validate a submission dict before it is written to disk.

Runs against the actual submission_template.json shape: a top-level object
with team / contact_email / model / answers, where `answers` maps scenario
id -> clause -> {status, actual, evidence_txn_id}.
"""
from __future__ import annotations
import math

REQUIRED_TOP_KEYS = ("team", "contact_email", "model", "answers")
REQUIRED_CELL_KEYS = ("status", "actual", "evidence_txn_id")
VALID_STATUS = {"COMPLIANT", "BREACH"}
EXPECTED_CELLS = 36


class SubmissionError(ValueError):
    pass


def validate_submission(sub: dict, template: dict) -> None:
    if not isinstance(sub, dict):
        raise SubmissionError("submission must be a JSON object")

    missing = [k for k in REQUIRED_TOP_KEYS if k not in sub]
    if missing:
        raise SubmissionError(f"submission is missing required key(s): {missing}")

    for key in ("team", "contact_email"):
        v = sub[key]
        if not isinstance(v, str) or not v.strip():
            raise SubmissionError(f"'{key}' must be a non-empty string, got {v!r}")

    answers = sub["answers"]
    if not isinstance(answers, dict):
        raise SubmissionError("'answers' must be an object keyed by scenario id")

    expected = template["answers"]
    cell_count = 0
    for sid, clauses in expected.items():
        if sid not in answers or not isinstance(answers[sid], dict):
            raise SubmissionError(f"submission is missing scenario '{sid}'")
        for clause in clauses:
            cell_count += 1
            if clause not in answers[sid]:
                raise SubmissionError(f"submission is missing cell {sid}/{clause}")
            cell = answers[sid][clause]
            if not isinstance(cell, dict):
                raise SubmissionError(f"cell {sid}/{clause} must be an object")

            missing_keys = [k for k in REQUIRED_CELL_KEYS if k not in cell]
            if missing_keys:
                raise SubmissionError(f"cell {sid}/{clause} is missing key(s): {missing_keys}")

            status = cell["status"]
            if status not in VALID_STATUS:
                raise SubmissionError(f"cell {sid}/{clause} has invalid status {status!r}")

            actual = cell["actual"]
            if isinstance(actual, bool) or not isinstance(actual, (int, float)):
                raise SubmissionError(f"cell {sid}/{clause} 'actual' must be a number, got {actual!r}")
            if not math.isfinite(actual):
                raise SubmissionError(f"cell {sid}/{clause} 'actual' must be finite, got {actual!r}")
            if actual <= 0:
                raise SubmissionError(f"cell {sid}/{clause} 'actual' must be greater than zero, got {actual!r}")

            ev = cell["evidence_txn_id"]
            if ev is not None and not isinstance(ev, str):
                raise SubmissionError(f"cell {sid}/{clause} 'evidence_txn_id' must be a string or null, got {ev!r}")

    if cell_count != EXPECTED_CELLS:
        raise SubmissionError(f"expected {EXPECTED_CELLS} cells, found {cell_count}")
