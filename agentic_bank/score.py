"""Local scorer — a faithful copy of the published rubric.

  status            0.50   exact string match
  actual            0.30   0.30 * max(0, 1 - relative_error/0.05)
  evidence_txn_id   0.20   exact match; when the key is null these 0.20 decay
                           on the same scale as `actual`
  wrong status  ->  whole cell scores 0
"""
from __future__ import annotations
import argparse
import json


def cell_score(pred: dict, key: dict) -> tuple[float, str]:
    if not isinstance(pred, dict) or pred.get("status") != key["status"]:
        return 0.0, "status"
    total, why = 0.5, ""

    kv = key.get("actual")
    pv = pred.get("actual")
    if kv in (None, 0) or not isinstance(pv, (int, float)):
        frac = 0.0
        why = "actual"
    else:
        err = abs(float(pv) - float(kv)) / abs(float(kv))
        frac = max(0.0, 1 - err / 0.05)
        if frac < 1:
            why = f"actual({err:.2%})"
    total += 0.30 * frac

    if key.get("evidence_txn_id") is None:
        total += 0.20 * frac                     # decays with `actual`
    elif pred.get("evidence_txn_id") == key["evidence_txn_id"]:
        total += 0.20
    else:
        why = (why + " evidence").strip()
    return total, why


def score(submission: dict, ground_truth: dict, show=True) -> float:
    gt = ground_truth["scenarios"]
    got = submission.get("answers", {})
    total = 0.0
    n = 0
    for sid in sorted(gt):
        for clause in sorted(gt[sid]["covenants"]):
            key = gt[sid]["covenants"][clause]
            pred = got.get(sid, {}).get(clause, {})
            s, why = cell_score(pred, key)
            total += s
            n += 1
            if show:
                mark = "OK " if s >= 0.999 else ("~  " if s > 0 else "XX ")
                print(f"{mark}{sid:4} {clause}  {s:.2f}  "
                      f"pred={pred.get('status'):9} {str(pred.get('actual')):>14} "
                      f"{str(pred.get('evidence_txn_id')):14} | "
                      f"key={key['status']:9} {key['actual']:>14,.2f} "
                      f"{str(key['evidence_txn_id']):14} {why}")
    if show:
        print(f"\nTOTAL {total:.2f} / {n}  ({total / n:.1%})")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("ground_truth")
    a = ap.parse_args()
    score(json.load(open(a.submission, encoding="utf-8")),
          json.load(open(a.ground_truth, encoding="utf-8")))


if __name__ == "__main__":
    main()
