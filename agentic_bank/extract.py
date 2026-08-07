"""Stage 3 — read the documents into machine-checkable structures.

The design principle: the model never does arithmetic and never decides
COMPLIANT/BREACH. It only translates prose into a small covenant DSL. All
numbers come from the ledger and all comparisons happen in evaluate.py, so a
covenant is either parsed correctly or it fails loudly — it can never be
"nearly right" because the model did mental maths.
"""
from __future__ import annotations
import os
import re
from .config import CATEGORIES
from .llm import LLMError

# --------------------------------------------------------------- schemas ----

COMPONENT = {
    "type": "object",
    "properties": {
        "categories": {"type": "array", "items": {"type": "string", "enum": CATEGORIES},
                       "description": "Ledger categories summed for this leg (absolute values)."},
        "related_party_only": {"type": "boolean",
                               "description": "Restrict to counterparties the KYC dossier marks as related."},
        "quarter": {"type": ["integer", "null"], "enum": [1, 2, 3, 4, None],
                    "description": "Restrict to one fiscal quarter, else null."},
        "largest_line_only": {"type": "boolean",
                              "description": "Take the largest single category total instead of the sum ('individual line ceiling')."},
        "include_addbacks": {"type": "boolean",
                             "description": "Add auditor-approved one-off add-backs (Adjusted EBITDA)."},
    },
    "required": ["categories"],
}

COVENANT_SCHEMA = {
    "type": "object",
    "properties": {
        "covenants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clause": {"type": "string", "description": "Exactly as printed, e.g. '6.2'."},
                    "title": {"type": "string"},
                    "kind": {"type": "string", "enum": ["ratio", "amount"]},
                    "direction": {"type": "string", "enum": ["max", "min"],
                                  "description": "'max' = breach when actual exceeds threshold."},
                    "threshold": {"type": "number", "description": "Positive number; strip $ and the trailing x."},
                    "numerator": COMPONENT,
                    "denominator": COMPONENT,
                    "period_start": {"type": "string"},
                    "period_end": {"type": "string"},
                    "springing_condition": {
                        "type": ["object", "null"],
                        "description": "Test applies only if this holds; otherwise COMPLIANT regardless.",
                        "properties": {
                            "component": COMPONENT,
                            "operator": {"type": "string", "enum": [">", ">=", "<", "<="]},
                            "value": {"type": "number"},
                        },
                    },
                    "notes": {"type": "string",
                              "description": "Any carve-out, cure or reclassification instruction, verbatim."},
                },
                "required": ["clause", "kind", "direction", "threshold", "numerator",
                             "period_start", "period_end"],
            },
        }
    },
    "required": ["covenants"],
}

KYC_SCHEMA = {
    "type": "object",
    "properties": {
        "account": {"type": "string"},
        "ownership_threshold_pct": {"type": "number",
                                    "description": "The percentage at or above which an entity counts as related. Read it from the sentence under the table; it is NOT always 20."},
        "holdings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string"},
                    "voting_pct": {"type": "number"},
                },
                "required": ["entity", "voting_pct"],
            },
        },
    },
    "required": ["ownership_threshold_pct", "holdings"],
}

ADJUSTMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_binding": {"type": "boolean",
                       "description": "False for interim/draft worksheets that a final report supersedes."},
        "defers_to_report": {"type": ["string", "null"],
                             "description": "Report number this document defers its conclusion to, if any."},
        "adjustments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string",
                             "enum": ["reclassify", "exclude", "include", "addback", "fx"]},
                    "txn_id": {"type": ["string", "null"]},
                    "amount": {"type": ["number", "null"],
                               "description": "Absolute amount, used to find the line when no txn id is quoted."},
                    "counterparty": {"type": ["string", "null"]},
                    "to_category": {"type": ["string", "null"], "enum": CATEGORIES + [None]},
                    "currency": {"type": ["string", "null"]},
                    "rate_to_usd": {"type": ["number", "null"],
                                    "description": "Derived rate when the note gives a foreign invoice and its USD settlement."},
                    "reason": {"type": "string"},
                },
                "required": ["kind", "reason"],
            },
        },
    },
    "required": ["is_binding", "adjustments"],
}

TXN_SCHEMA = {
    "type": "object",
    "properties": {
        "transactions": {
            "type": "array",
            "description": "Exactly one entry per input row, same order as the input, no omissions.",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "The txn_id, copied exactly as given."},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "filler": {"type": "boolean",
                               "description": "True if this row is a filler/separator row rather "
                                               "than a genuine transaction (see system prompt)."},
                },
                "required": ["id", "category", "filler"],
            },
        }
    },
    "required": ["transactions"],
}

# --------------------------------------------------------------- prompts ----

COVENANT_SYSTEM = """You convert Kazakh/Russian bank loan covenants into a strict machine schema.

Rules you must not break:
- Copy the clause number exactly as printed ("Пункт 6.2" -> "6.2").
- `threshold` is a plain positive number: "$1,500,000.00" -> 1500000, "2.00x" -> 2.
- `direction` is "max" when the borrower must NOT exceed the figure ("не допускать,
  чтобы ... превышал"), "min" when it must stay at or above it ("не менее", "не ниже").
- kind="ratio" needs both numerator and denominator. kind="amount" needs numerator only —
  set denominator to null for an amount covenant (never invent one).
- If there is nothing to say, set `notes` to null rather than an empty explanation.
- Map economic concepts onto the fixed category list:
    Выручка -> revenue
    Операционные расходы -> opex
    Капитальные затраты -> capex
    арендные платежи / лизинг -> lease
    расходы на оплату труда / персонал -> payroll
    коммунальные услуги -> utilities
    Процентные расходы -> interest_expense
    поступления по финансированию -> financing_inflow
    налоги -> tax, страхование -> insurance
  EBITDA = Выручка минус Операционные расходы: numerator {revenue}, denominator {opex}
  is WRONG. Express EBITDA as numerator categories ["revenue"] with
  `subtract` handled by the caller — instead, when a leg is "EBITDA", set its
  categories to ["revenue","opex"] and say EBITDA in `notes`; the evaluator nets
  inflow categories against expense categories automatically.
- "Скорректированная EBITDA" additionally sets include_addbacks=true.
- A test that applies "только при условии, что X превышает Y" is a
  springing_condition, not part of the ratio.
- "отдельными статьями ... признаются по отдельности, а не в совокупности" and
  "по наибольшей из указанных сумм" mean largest_line_only=true.
- "за четвёртый квартал" sets quarter=4 on that leg.
- Related-party / аффилированные лица legs set related_party_only=true.
Return every clause under Статья 6, and only those."""

KYC_SYSTEM = """Extract the related-party test from a KYC dossier.

The ownership cut-off is stated in prose beneath the holdings table and VARIES
between borrowers (20%, 25%, 40% ...). Read it; never assume. Copy entity names
exactly, including punctuation such as 'L.L.P.' or quotation marks, because the
ledger writes the same company slightly differently and the matcher needs the
original spelling."""

ADJUSTMENT_SYSTEM = """Extract auditor conclusions that change how ledger lines are counted.

Critical precedence rules:
- An "промежуточная ведомость" / "ПРОЕКТ" / worksheet that says it is replaced by
  the final agreed-procedures report is NOT binding: set is_binding=false and
  still list what it claimed, so the caller can prove it was excluded.
- Notes to the accounts that say the conclusion "изложен в отчёте № AR-xxxx и
  здесь не повторяется" carry no adjustment themselves: set defers_to_report.
- A cut-off note ("услуги оказаны в период с 2026-...") is kind="exclude": the
  line belongs to a different covenant period.
- A reclassification names the target line item -> kind="reclassify" + to_category.
- If a note gives a foreign-currency invoice and the USD amount that settled it,
  emit kind="fx" with rate_to_usd = usd / foreign.
- Final reports often identify a line by amount and counterparty rather than by
  txn id. Copy both; leave txn_id null."""

TXN_SYSTEM = """You are classifying corporate bank ledger lines into accounting line items.

Judge by the DESCRIPTION only. Counterparty names in this ledger are randomised
noise: a payment to "Foxridge Stationery" described as a corporate income tax
instalment is tax, not stationery.

Category guide:
- revenue: core trading income, phrased "<activity> sales settlement".
- opex: core operating/maintenance of the productive asset — "servicing and
  operating costs", "operating and maintenance expenses", servicing contracts.
- capex: "Purchase of ... equipment", construction and major repair works.
- lease: rent and lease of premises, land, yards, masts, vehicles.
- payroll: wages, bonuses, staff payroll runs.
- utilities: electricity, water, gas, heating, waste water, metering.
- interest_expense / interest_income: interest paid / earned.
- tax, insurance, marketing, telecom, professional_fees (advisory, consulting,
  management retainers): self-explanatory.
- financing_inflow: loan or facility drawdowns.
- credit_refund: money flowing BACK — refunds, rebates, credits, deposits
  released, overpayments returned, accrual reversals. A positive amount whose
  description is an expense word is almost always credit_refund, NOT revenue.

Note the sign convention: negative = money out, positive = money in. Classify
every line you are given, and return exactly one entry per input row, using
its id exactly as given — never omit a row, invent one, or merge two rows
into one entry.

Some rows are not real transactions at all: they are FILLER — visual
separators, section headings, decorative or blank rows, service/boilerplate
text, or rows that simply describe no economic event. Set filler=true for
those rows and filler=false for every genuine transaction line, based on
the overall content and structure of the row. Do NOT decide filler status by
checking for one specific character or separator symbol (e.g. a dash) — that
signal is unreliable and dataset-specific. Instead weigh the row as a whole:
does it have a plausible amount, a valid date, meaningful non-empty fields,
and description text long enough to describe an actual economic event? Still
assign your best-guess category to filler rows as well; every row needs both
fields."""


# ------------------------------------------------------------- extractors ---

def covenant_section(agreement_text: str) -> str:
    """Slice Статья 6 out of a ~48k character agreement to keep the prompt tight."""
    flat = agreement_text
    m = re.search(r"Стать[яи]\s*6\s*[—\-–]\s*Финансовые ковенант", flat)
    if not m:
        m = re.search(r"Пункт\s*6\.1", flat)
    if not m:
        return flat[:20000]
    start = m.start()
    n = re.search(r"Стать[яи]\s*7\s*[—\-–]", flat[start:])
    end = start + (n.start() if n else 12000)
    return flat[start:end]


def _covenant_semantic_errors(parsed: dict) -> list[str]:
    """Business-rule check the generic JSON-Schema subset can't express:
    `denominator` is only genuinely optional for kind="amount" (evaluate.metric()
    never reads it there). A kind="ratio" covenant divides by it, so null/absent
    is never a valid representation for one — unlike `notes`, which no code
    anywhere reads, or an amount covenant's denominator, which is inert."""
    errors = []
    for i, c in enumerate(parsed.get("covenants") or []):
        if c.get("kind") == "ratio" and c.get("denominator") is None:
            errors.append(f"$.covenants[{i}].denominator: null on a kind=\"ratio\" covenant "
                           f"(a ratio needs a real denominator; only kind=\"amount\" may omit it)")
    return errors


def extract_covenants(llm, agreement_text: str) -> list[dict]:
    body = covenant_section(agreement_text)
    # 2048: each clause nests numerator/denominator/springing_condition
    # objects plus a free-text `notes` field, and Статья 6 typically holds
    # several clauses — the small 1024 budget is genuinely tight here.
    out = llm.json_call(COVENANT_SYSTEM, body, COVENANT_SCHEMA, name="covenants", max_tokens=2048,
                         extra_validate=_covenant_semantic_errors)
    return out.get("covenants", [])


def extract_kyc(llm, kyc_text: str) -> dict:
    # 1024: a threshold percentage plus a short holdings list — small output.
    return llm.json_call(KYC_SYSTEM, kyc_text[:20000], KYC_SCHEMA, name="kyc", max_tokens=1024)


def extract_adjustments(llm, doc_text: str) -> dict:
    # 1024: a handful of adjustments per document, each a few short fields
    # plus a one-line `reason`.
    return llm.json_call(ADJUSTMENT_SYSTEM, doc_text[:20000], ADJUSTMENT_SCHEMA, name="adjustments",
                          max_tokens=1024)


DEFAULT_BATCH_SIZE = 50


def _batch_size() -> int:
    n = int(os.environ.get("AB_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
    if n < 1:
        raise ValueError(f"AB_BATCH_SIZE must be a positive integer, got {n}")
    return n


def _vote_count(llm) -> int:
    votes = getattr(llm.cfg, "self_consistency", 1)
    if votes < 1 or votes % 2 == 0:
        raise ValueError(f"AB_VOTES must be a positive odd integer, got {votes}")
    return votes


def _validate_ballot(out: dict, expected_ids: set) -> dict[str, dict]:
    """Validate one batched classifier response against the input row ids.

    Every input id must appear exactly once, with a valid category and a
    boolean filler flag. Unknown ids, duplicates, bad categories/types, or
    missing rows are all treated as a broken response — raise rather than
    silently patch it, since a partial ballot would corrupt the vote.
    """
    items = out.get("transactions")
    if not isinstance(items, list):
        raise LLMError("classifier response is missing a 'transactions' array")

    seen: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            raise LLMError(f"classifier response contains a non-object transaction entry: {item!r}")
        tid = item.get("id")
        if not isinstance(tid, str) or tid not in expected_ids:
            raise LLMError(f"classifier response contains an unknown transaction id: {tid!r}")
        if tid in seen:
            raise LLMError(f"classifier response contains a duplicate transaction id: {tid!r}")
        category = item.get("category")
        if category not in CATEGORIES:
            raise LLMError(f"classifier returned an invalid category {category!r} for {tid}")
        filler = item.get("filler")
        if not isinstance(filler, bool):
            raise LLMError(f"classifier returned a non-boolean 'filler' for {tid}: {filler!r}")
        seen[tid] = {"category": category, "is_filler": filler}

    missing = expected_ids - seen.keys()
    if missing:
        raise LLMError(f"classifier response is missing {len(missing)} transaction id(s): "
                        f"{sorted(missing)[:10]}")
    return seen


def _majority_category(votes: list[str]) -> str:
    counts: dict[str, int] = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    winner, best = max(counts.items(), key=lambda kv: kv[1])
    if best * 2 > len(votes):
        return winner
    return votes[0]  # no strict majority: keep the first vote, deterministic tie-break


def _majority_bool(votes: list[bool]) -> bool:
    yes = sum(1 for v in votes if v)
    return yes * 2 > len(votes)


def _amt(v) -> float:
    """The ledger ships a handful of rows with an EMPTY amount — not
    corrupt data: the real figure is disclosed in a treasury memo or
    auditor note and patched onto the transaction later via a `set_amount`
    adjustment in run.build_context() (see the identical contract in
    rules._amt / run._ledger_amount). The row itself is real and must still
    be classified and kept, never dropped, so 0.0 is a safe placeholder
    here — it only steers the classifier's own sign-based heuristics
    (`expense description + positive amount => credit_refund`), it is
    never used as the covenant-arithmetic value."""
    try:
        return float(str(v).strip())
    except ValueError:
        return 0.0


def classify_transactions(llm, rows: list[dict], batch: int | None = None) -> dict[str, dict]:
    """Classify a borrower's ledger with one batched call per chunk of rows.

    One call classifies up to AB_BATCH_SIZE (default 50) rows at once — a
    whole scenario's ledger fits in a single call whenever it's that small.
    Majority voting (AB_VOTES independent calls) is applied per batch, not
    per transaction: each vote classifies the whole chunk in one call, and
    the majority is then taken independently for every transaction id.
    """
    votes = _vote_count(llm)
    batch_size = batch or _batch_size()

    result: dict[str, dict] = {}
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        expected_ids = {r["txn_id"] for r in chunk}
        lines = "\n".join(
            f"{r['txn_id']} | {r['date']} | {_amt(r['amount']):.2f} {r['currency']} | "
            f"{r['counterparty']} | {r['description']}"
            for r in chunk
        )

        ballots: list[dict[str, dict]] = []
        for v in range(votes):
            pass_user = lines if votes == 1 else f"{lines}\n\n[self-consistency pass {v + 1}/{votes}]"
            # 2048: one {id, category, filler} entry per row, up to
            # AB_BATCH_SIZE (default 50) rows in a single response — 1024
            # would be tight at the default batch size.
            out = llm.json_call(TXN_SYSTEM, pass_user, TXN_SCHEMA, name="classify", max_tokens=2048)
            ballots.append(_validate_ballot(out, expected_ids))

        for tid in expected_ids:
            cat_votes = [b[tid]["category"] for b in ballots]
            filler_votes = [b[tid]["is_filler"] for b in ballots]
            category = _majority_category(cat_votes)
            is_filler = _majority_bool(filler_votes)
            if votes > 1:
                print(f"VOTE txn={tid} categories={cat_votes} filler={filler_votes} "
                      f"winner={category} filler={is_filler}")
            result[tid] = {"category": category, "is_filler": is_filler}
    return result
