"""Stage 4 — deterministic arithmetic. No model involved past this point.

`actual` is always reported as a positive number, in dollars for amount tests
and as a bare multiple for ratio tests, per the scoring rules.
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .config import EXPENSE_CATEGORIES, INFLOW_CATEGORIES


@dataclass
class Txn:
    txn_id: str
    date: str
    counterparty: str
    description: str
    amount: float
    currency: str
    category: str = "other"
    related_party: bool = False
    unrestricted_transfer: bool = False
    excluded: bool = False          # auditor cut-off: belongs to another period
    excluded_reason: str = ""
    original_category: str | None = None
    is_filler: bool = False         # set by the classification stage (LLM prompt or rules heuristic)

    @property
    def core(self) -> bool:
        """True for a genuine ledger line; false for filler/separator rows.

        The ledger mixes each borrower's real line items with filler rows.
        Aggregating filler inflates every denominator and turns compliant
        ratios into breaches, so filler must be excluded here. Detection
        itself happens upstream, during transaction classification.
        """
        return not self.is_filler

    @property
    def quarter(self) -> int:
        return (int(self.date[5:7]) - 1) // 3 + 1

    def usd(self, fx: dict[str, float]) -> float:
        return self.amount * fx.get(self.currency, 1.0)


@dataclass
class Context:
    txns: list[Txn]
    fx: dict[str, float] = field(default_factory=lambda: {"USD": 1.0})
    addbacks: list[float] = field(default_factory=list)
    # Amounts that are NOT in the ledger and must come from a document:
    # group capex from the parent's consolidated accounts, an undrawn severance
    # provision disclosed in the notes, accrued-but-unpaid tax from treasury
    # records, assets transferred to an unrestricted subsidiary, and so on.
    extras: dict[str, float] = field(default_factory=dict)


def leg_value(ctx: Context, leg: dict, skip: str | None = None) -> float:
    """Value of one side of a covenant.

    Mixing an inflow category with an expense category in the same leg means
    'net' (that is how EBITDA = revenue - opex is expressed).
    """
    cats = set(leg.get("categories") or [])
    extra = sum(ctx.extras.get(name, 0.0) for name in (leg.get("extras") or []))
    if not cats:
        return extra
    q = leg.get("quarter")
    rp_only = bool(leg.get("related_party_only"))
    ut_only = bool(leg.get("unrestricted_only"))

    per_cat: dict[str, float] = {c: 0.0 for c in cats}
    for t in ctx.txns:
        if t.excluded or t.txn_id == skip or t.category not in cats or not t.core:
            continue
        if q and t.quarter != q:
            continue
        if rp_only and not t.related_party:
            continue
        if ut_only and not t.unrestricted_transfer:
            continue
        amt = t.usd(ctx.fx)
        if t.category in EXPENSE_CATEGORIES:
            if amt < 0:                       # ignore credits sitting in an expense line
                per_cat[t.category] += -amt
        elif t.category in INFLOW_CATEGORIES:
            if amt > 0:
                per_cat[t.category] += amt
        else:
            per_cat[t.category] += abs(amt)

    if leg.get("largest_line_only"):
        return (max(per_cat.values()) if per_cat else 0.0) + extra

    inflow = sum(v for c, v in per_cat.items() if c in INFLOW_CATEGORIES)
    expense = sum(v for c, v in per_cat.items() if c in EXPENSE_CATEGORIES)
    other = sum(v for c, v in per_cat.items()
                if c not in INFLOW_CATEGORIES and c not in EXPENSE_CATEGORIES)

    value = (inflow - expense + other) if (inflow and expense) else (inflow + expense + other)
    value += extra
    if leg.get("subtract_largest_of"):
        # "Revenue less the LARGER of payroll and tax" — the smaller is ignored.
        legs = [{"categories": [c]} for c in leg["subtract_largest_of"]]
        value -= max(leg_value(ctx, sub, skip) for sub in legs)
    if leg.get("include_addbacks"):
        value += sum(ctx.addbacks)
    return value


def metric(ctx: Context, spec: dict, skip: str | None = None) -> float | None:
    num = leg_value(ctx, spec["numerator"], skip)
    if spec["kind"] == "amount":
        return abs(num)
    den = leg_value(ctx, spec.get("denominator") or {"categories": []}, skip)
    if den == 0:
        return None
    return abs(num / den)


def springing_applies(ctx: Context, spec: dict, skip: str | None = None) -> bool:
    cond = spec.get("springing_condition")
    if not cond:
        return True
    v = leg_value(ctx, cond["component"], skip)
    op, ref = cond["operator"], cond["value"]
    return {">": v > ref, ">=": v >= ref, "<": v < ref, "<=": v <= ref}[op]


def verdict(ctx: Context, spec: dict, skip: str | None = None) -> tuple[str, float | None]:
    value = metric(ctx, spec, skip)
    if value is None:
        return "COMPLIANT", None
    if not springing_applies(ctx, spec, skip):
        return "COMPLIANT", value          # actual is still the real measure
    thr = float(spec["threshold"])
    if spec["direction"] == "max":
        return ("BREACH" if value > thr else "COMPLIANT"), value
    return ("BREACH" if value < thr else "COMPLIANT"), value


def find_evidence(ctx: Context, spec: dict, status: str, touched: set[str]) -> str | None:
    """The single transaction whose reclassification, inclusion or exclusion
    produced the breach — remove it and the verdict flips.

    A line that merely contributes to a total is not evidence, so if several
    different removals would flip the verdict we return null rather than guess.
    `touched` (lines an auditor or the KYC dossier acted on) breaks ties: those
    are the lines whose *treatment*, not whose size, drove the result.
    """
    if status != "BREACH":
        return None
    flippers = []
    for t in ctx.txns:
        if t.excluded:
            continue
        st, _ = verdict(ctx, spec, skip=t.txn_id)
        if st != status:
            flippers.append(t.txn_id)
    if len(flippers) == 1:
        return flippers[0]
    inside = [f for f in flippers if f in touched]
    if len(inside) == 1:
        return inside[0]
    return None


def round2(x: float | None) -> float | None:
    return None if x is None else round(float(x) + 0.0, 2)


def final_actual(value: float | None) -> float | None:
    """The submission's reported `actual` (see module docstring: "always...
    a positive number"): a genuine measurement, or null when there is
    nothing meaningful to report. `metric()` already returns None for a
    ratio whose denominator is zero; this folds a computed literal zero
    into that same null representation too — the scoring rubric
    (score.py's `cell_score`) already treats a zero actual identically to
    a null one, so a bare 0.0 is never a distinct, validly-reported value
    here, only ever a sign that nothing was there to measure."""
    rounded = round2(value)
    return None if rounded == 0 else rounded
