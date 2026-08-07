"""Stage 5 — orchestration. `python -m agentic_bank.run --dataset ... --out submission.json`"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import unicodedata
from collections import defaultdict

from . import covenant_rules, extract, rules
from .config import LLMConfig, Paths, Submission
from .evaluate import Context, Txn, find_evidence, round2, verdict
from .ingest import ingest_all
from .route import build_index, select_sources
from .extract import covenant_section
from .validate import SubmissionError, validate_submission


# ------------------------------------------------------------- ledger -------
def load_ledger(path: str):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    by_scenario = defaultdict(list)
    acc_to_scenario: dict[str, str] = {}
    for r in rows:
        sid = r["txn_id"].split("-")[1]
        by_scenario[sid].append(r)
        acc_to_scenario[r["account_id"]] = sid
    return by_scenario, acc_to_scenario


# --------------------------------------------------- related party matching --
def _canon(name: str) -> frozenset:
    """Comparable token set for a counterparty.

    Only pure legal forms are dropped. Descriptive words are kept: 'Taraz
    Holding Group LLP' and 'Taraz Kiln Services LLP' are different entities and
    a matcher that reduces both to 'taraz' will silently invent related-party
    payments — the fastest way to turn a compliant covenant into a breach.
    """
    n = unicodedata.normalize("NFKD", name).lower()
    n = re.sub(r"\(.*?\)", " ", n)                       # "(Ekibastuz block B)"
    n = re.sub(r"[^a-z0-9 ]", " ", n)                     # "L.L.P." -> "l l p"
    tokens = [t for t in n.split() if len(t) > 1]
    legal = {"llp", "llc", "ltd", "jsc", "inc", "corp", "co", "plc", "gmbh", "lp"}
    return frozenset(t for t in tokens if t not in legal)


def related_parties(kyc: dict) -> list[str]:
    thr = float(kyc.get("ownership_threshold_pct", 20.0))
    return [h["entity"] for h in kyc.get("holdings", []) if float(h["voting_pct"]) >= thr]


def is_related(counterparty: str, names: list[str]) -> bool:
    c = _canon(counterparty)
    if not c:
        return False
    for n in names:
        k = _canon(n)
        if len(k) >= 2 and k <= c:
            return True
        if k and k == c:
            return True
    return False


# --------------------------------------------------------------- pipeline ---
def _ledger_amount(v) -> float:
    try:
        return float(str(v).strip())
    except ValueError:
        return 0.0


def build_context(rows, kyc, categories, adjustments, extras) -> tuple[Context, set[str]]:
    txns = [
        Txn(txn_id=r["txn_id"], date=r["date"], counterparty=r["counterparty"],
            description=r["description"], amount=_ledger_amount(r["amount"]),
            currency=r["currency"])
        for r in rows
    ]
    for t in txns:
        c = categories.get(t.txn_id) or {}
        t.category = c.get("category", "other")
        t.is_filler = bool(c.get("is_filler", False))
    index = {t.txn_id: t for t in txns}
    names = related_parties(kyc) if kyc else []
    for t in txns:
        t.related_party = is_related(t.counterparty, names)

    ctx = Context(txns=txns, extras=extras or {})
    touched: set[str] = {t.txn_id for t in txns if t.related_party}

    rejected: set[str] = set()
    for doc in adjustments:
        for a in doc.get("adjustments", []):
            if a["kind"] == "rejected" and a.get("txn_id"):
                rejected.add(a["txn_id"])

    for doc in adjustments:
        if not doc.get("is_binding", True):
            continue                                   # superseded draft worksheet
        for a in doc.get("adjustments", []):
            if a["kind"] == "rejected":
                continue
            if a["kind"] == "disclosure":
                ctx.extras[a["name"]] = ctx.extras.get(a["name"], 0.0) + abs(float(a["amount"]))
                continue
            if a["kind"] == "fx" and a.get("rate_to_usd"):
                ctx.fx[a.get("currency") or "EUR"] = float(a["rate_to_usd"])
                continue
            target = _locate(a, index, txns)
            if target is None:
                continue
            if target.txn_id in rejected and a["kind"] == "reclassify":
                continue                               # auditor declined this one
            touched.add(target.txn_id)
            if a["kind"] == "set_amount":
                sign = -1 if "расход" in a.get("reason", "") or target.amount <= 0 else 1
                target.amount = sign * abs(float(a["amount"]))
            elif a["kind"] == "exclude":
                target.excluded, target.excluded_reason = True, a.get("reason", "")
            elif a["kind"] == "reclassify" and a.get("to_category"):
                target.original_category = target.category
                target.category = a["to_category"]
            elif a["kind"] == "addback":
                # A one-off item added back to EBITDA is by construction an
                # operating cost: it was deducted inside EBITDA before the
                # auditor added it back.
                target.category = "opex"
                ctx.addbacks.append(abs(target.amount))
    return ctx, touched


def _locate(a: dict, index: dict, txns: list[Txn]) -> Txn | None:
    if a.get("txn_id") and a["txn_id"] in index:
        return index[a["txn_id"]]
    amt = a.get("amount")
    if amt:
        hits = [t for t in txns if abs(abs(t.amount) - abs(float(amt))) < 0.01]
        if len(hits) == 1:
            return hits[0]
        if hits and a.get("counterparty"):
            c = _canon(a["counterparty"])
            for t in hits:
                if c and (c in _canon(t.counterparty) or _canon(t.counterparty) in c):
                    return t
    return None


def solve_scenario(scenario, rows, docs, backend, llm=None, verbose=False):
    src = select_sources(docs, scenario)
    agreement = src["credit_agreement_current"]
    if not agreement:
        agreement = src["credit_agreement_superseded"]   # last resort, flagged below
    section = covenant_section(agreement[0].text) if agreement else ""

    kyc_docs = src["kyc"]
    audit_docs = (src["audit_notes"] + src["audit_agreed_procedures"]
                  + src["audit_draft_worksheet"] + src["treasury_memo"])

    extras: dict[str, float] = {}
    for cd in src.get("consolidated_accounts", []):
        gc = rules.group_capex(cd.text)
        if gc:
            extras["group_capex"] = gc

    if backend == "llm":
        specs = extract.extract_covenants(llm, section)
        kyc = extract.extract_kyc(llm, kyc_docs[0].text) if kyc_docs else {}
        adjustments = [extract.extract_adjustments(llm, d.text) for d in audit_docs]
        cats = extract.classify_transactions(llm, rows)
    else:
        specs = covenant_rules.extract_covenants(section)
        kyc = rules.extract_kyc(kyc_docs[0].text) if kyc_docs else {}
        adjustments = [rules.extract_adjustments(d.text, d.doc_type) for d in audit_docs]
        cats = rules.classify_transactions(rows)

    unrestricted = rules.unrestricted_subsidiaries(kyc_docs[0].text) if kyc_docs else []
    ctx, touched = build_context(rows, kyc, cats, adjustments, extras)
    if unrestricted:
        # capital assets moved to a subsidiary outside the security perimeter
        for t in ctx.txns:
            t.unrestricted_transfer = (t.category == "capex"
                                       and is_related(t.counterparty, unrestricted))
        touched |= {t.txn_id for t in ctx.txns if t.unrestricted_transfer}

    answers = {}
    for spec in specs:
        status, value = verdict(ctx, spec)
        ev = find_evidence(ctx, spec, status, touched)
        answers[spec["clause"]] = {"status": status, "actual": round2(value),
                                   "evidence_txn_id": ev}
        if verbose:
            print(f"  {scenario} {spec['clause']:4} {status:9} {value!r:>16} ev={ev}")
    return answers, ctx, specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="submission.json")
    ap.add_argument("--backend", choices=["llm", "rules"], default="llm")
    ap.add_argument("--work", default=".cache")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--vision-cache", default=None,
                    help="Directory of pre-transcribed image pages: {doc_id}.p{n}.txt")
    args = ap.parse_args()

    paths = Paths(dataset=args.dataset, work=args.work)
    llm = None
    if args.backend == "llm":
        from .llm import LLM
        cfg = LLMConfig()
        cfg.verbose = args.verbose
        if not cfg.api_key:
            env = "GROQ_API_KEY" if cfg.provider == "groq" else "GEMINI_API_KEY"
            raise SystemExit(f"{env} is not set (LLM_PROVIDER={cfg.provider!r})")
        # cfg.vision_api_key is checked lazily, only if a scanned page
        # actually needs OCR — most runs never call vision() at all (see
        # ingest.py), and requiring a second key up front would make the
        # common case harder to run than before.
        llm = LLM(cfg, paths.llm_cache)

    texts = ingest_all(paths.documents, paths.text_cache, llm, args.vision_cache)

    if llm is not None and os.environ.get("AB_BYPASS_VISION_CACHE") == "1":
        expected = 4
        if llm.vision_calls != expected:
            raise SystemExit(
                f"AB_BYPASS_VISION_CACHE=1 verification failed: expected exactly "
                f"{expected} real VISION_API_CALL requests, got {llm.vision_calls}."
            )

    by_scenario, acc_to_scenario = load_ledger(paths.ledger)
    docs = build_index(texts, acc_to_scenario)

    template = json.load(open(paths.template, encoding="utf-8"))
    sub = Submission()
    if llm is not None and "AB_MODEL" not in os.environ:
        sub.model = cfg.model  # reflect the actually-configured model, not a stale default
    out = {"team": sub.team, "contact_email": sub.contact_email, "model": sub.model,
           "answers": {}}

    for scenario, cells in template["answers"].items():
        answers, _, _ = solve_scenario(scenario, by_scenario.get(scenario, []), docs,
                                       args.backend, llm, args.verbose)
        out["answers"][scenario] = {
            clause: answers.get(clause, {"status": "COMPLIANT", "actual": 0.0,
                                         "evidence_txn_id": None})
            for clause in cells
        }

    try:
        validate_submission(out, template)
    except SubmissionError as e:
        if args.backend == "llm":
            # This is the file meant for grading: never write an invalid one.
            raise SystemExit(f"submission failed validation, not writing {args.out}: {e}")
        # `rules` is an explicitly approximate deterministic cross-check
        # (see rules.py / covenant_rules.py docstrings), not a real
        # submission, and it is expected to sometimes misfire. Warn instead
        # of aborting so it still gets written for diff comparison against
        # the llm backend.
        print(f"WARNING: {args.out} failed strict submission validation ({e}); "
              f"writing anyway since --backend rules is a diagnostic baseline, not a graded submission.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"wrote {args.out}" + (f" ({llm.calls} model calls)" if llm else ""))


if __name__ == "__main__":
    main()
