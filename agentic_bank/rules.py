"""A deterministic backend that produces the same structures as extract.py.

Purpose: run and regression-test the whole pipeline without network access,
and act as a cross-check on the model. When the two backends disagree on a
transaction's category, that transaction is worth a human's attention — the
`--backend both` mode prints exactly those.
"""
from __future__ import annotations
import re

# ------------------------------------------------- transaction categories ---
# Order matters: the first pattern that matches wins.
TXN_RULES: list[tuple[str, str]] = [
    (r"sales settlement", "revenue"),
    (r"drawdown", "financing_inflow"),
    (r"\b(refund|rebate|returned|reversal|credit received|recovered|released|"
     r"reimbursement|overbilling|credit note|sweep back|adjustment credit|"
     r"recovery|experience refund|premium return|true-?up)\b", "credit_refund"),
    (r"insurance|indemnity", "insurance"),
    (r"(servicing and operating|operating and maintenance|servicing contract|"
     r"regeneration servicing|risk survey servicing|remediation|repair works|"
     r"dispute arbitration|dredg|silt \w+ and clearance|clearance works)", "opex"),
    # "transfer of X equipment to subsidiary" is capital expenditure too: the
    # asset leaves the borrower, and the covenant on unrestricted subsidiaries
    # measures exactly these lines against total capex.
    (r"(purchase of .*equipment|transfer of .*equipment|construction of)", "capex"),
    (r"payroll|staff wages", "payroll"),
    (r"(electricity|water charge|water supply|municipal water|waste water|"
     r"district heating|natural gas|utility|metering|compressed air|"
     r"network capacity charge|standby generator)", "utilities"),
    (r"interest income|interest credited|interest on (term deposit|escrow|"
     r"treasury|current account)", "interest_income"),
    (r"interest", "interest_expense"),
    (r"(lease|rent\b|rent for|ground lease|site hire)", "lease"),
    (r"(tax|duty|levy|customs)", "tax"),
    (r"(marketing|ad campaign|advertis|media buy|sponsorship|exhibition|"
     r"newsletter|collateral|photography|press insertion|brand)", "marketing"),
    (r"(telecom|leased line|mobile fleet)", "telecom"),
    (r"(advisory|consult|retainer|management fee)", "professional_fees"),
]


def classify_txn(description: str, amount: float) -> str:
    d = description.lower()
    for pattern, category in TXN_RULES:
        if re.search(pattern, d):
            cat = category
            break
    else:
        cat = "other"
    # A positive amount on an expense-shaped line is money coming back.
    from .config import EXPENSE_CATEGORIES
    if amount > 0 and cat in EXPENSE_CATEGORIES:
        cat = "credit_refund"
    return cat


def _amt(v) -> float:
    """The ledger ships rows with an EMPTY amount. They are not corrupt: the
    real figure is disclosed in a treasury memo or an auditor note, and the
    covenant that needs it is exactly the one those rows belong to."""
    try:
        return float(str(v).strip())
    except ValueError:
        return 0.0


def _is_filler(description: str) -> bool:
    """The ledger mixes each borrower's real line items with filler rows.

    The filler always carries a qualifier after an em dash — a branch, a
    month, a quarter ("Depot yard rent — Pavlodar block, Q4 2025"). The
    borrower's actual line items never do ("Warehouse land lease payments
    2025"). This deterministic heuristic is specific to the rules backend;
    the LLM backend instead judges filler rows from the prompt in extract.py.
    """
    return "—" in description or " - " in description


def classify_transactions(rows: list[dict]) -> dict[str, dict]:
    return {
        r["txn_id"]: {
            "category": classify_txn(r["description"], _amt(r["amount"])),
            "is_filler": _is_filler(r["description"]),
        }
        for r in rows
    }


# ------------------------------------------------------------------- KYC ----
PCT_ROW = re.compile(r"^\s*(?P<name>[^|\n]{3,70}?)\s{2,}(?P<pct>\d{1,3}(?:\.\d+)?)\s*%\s*$", re.M)
PCT_ROW_ALT = re.compile(r"(?P<name>[A-Z\"«][^\n:]{3,70}?)[:\s]{2,}(?P<pct>\d{1,3}(?:\.\d+)?)\s*%")
THRESHOLD = re.compile(r"владеет\s+(\d{1,3}(?:\.\d+)?)\s*%?\s*и более", re.I)
THRESHOLD_ALT = re.compile(r"Группа владеет\s+(\d{1,3}(?:\.\d+)?)", re.I)


def _ownership_block(text: str) -> str:
    """Only the beneficial-ownership table. A KYC file can carry a SECOND
    percentage table (share of subsidiary assets pledged as security); reading
    both tables as ownership invents related parties out of subsidiaries."""
    low = text.lower()
    start = low.find("бенефициарное владение")
    if start < 0:
        return text
    for stop_word in ("идентификация и проверка", "обеспечительное покрытие",
                      "проверка по санкционным"):
        stop = low.find(stop_word, start)
        if stop > 0:
            return text[start:stop]
    return text[start:]


def extract_kyc(text: str) -> dict:
    text = _ownership_block(text)
    thr = 20.0
    m = THRESHOLD.search(text) or THRESHOLD_ALT.search(text)
    if m:
        thr = float(m.group(1))
    else:
        m2 = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%?\s*и более голосующих прав", text)
        if m2:
            thr = float(m2.group(1))
    holdings = []
    seen = set()
    for rx in (PCT_ROW, PCT_ROW_ALT):
        for mm in rx.finditer(text):
            name = mm.group("name").strip(" .·|")
            if not name or name.lower().startswith(("доля", "организация")):
                continue
            if name in seen:
                continue
            seen.add(name)
            holdings.append({"entity": name, "voting_pct": float(mm.group("pct"))})
    return {"ownership_threshold_pct": thr, "holdings": holdings}


# ----------------------------------------------------------- adjustments ----
MONEY = r"\$?\s?([\d\s,]{4,20}\.\d{2})"


def _money(s: str) -> float:
    return float(re.sub(r"[^\d.]", "", s))


def extract_adjustments(text: str, doc_type: str) -> dict:
    # Documents wrap mid-sentence; every pattern below must see one line.
    flat = re.sub(r"\s+", " ", text)
    binding = doc_type != "audit_draft_worksheet"
    defers = None
    m = re.search(r"отч[её]те?\s+о выполнении согласованных процедур\s*№?\s*(AR-\d{4}-\d{4})", flat)
    if m:
        defers = m.group(1)
    adj: list[dict] = []

    # cut-off: services rendered in another covenant period
    for mm in re.finditer(
        r"Операция\s+(TXN-[A-Z0-9\-]+)[^.]*?относится к услугам, оказанным в период с (\d{4})",
        flat,
    ):
        if mm.group(2) != "2025":
            adj.append({"kind": "exclude", "txn_id": mm.group(1), "reason": "cut-off to another period"})

    for mm in re.finditer(
        r"Операция\s+(TXN-[A-Z0-9\-]+)[^.]{0,120}?исключена из ковенантного периода", flat):
        adj.append({"kind": "exclude", "txn_id": mm.group(1), "reason": "excluded from the period"})

    # reclassification identified by txn id
    for mm in re.finditer(
        r"(TXN-[A-Z0-9\-]+)[^.]{0,200}?переклассифицирован[а-я]* .{0,60}?как ([А-Яа-яЁё ]{4,40})", flat
    ):
        adj.append({"kind": "reclassify", "txn_id": mm.group(1),
                    "to_category": _map_line_item(mm.group(2)), "reason": mm.group(2).strip()})

    # reclassification identified by amount + counterparty (final AP reports)
    for mm in re.finditer(
        r"Сумма в размере " + MONEY + r"[^.]{0,120}?контрагенту\s+([^,]{3,60}),"
        r"[^.]{0,200}?переклассифицирован[а-я]*[^.]{0,80}?как ([А-Яа-яЁё ]{4,40})", flat
    ):
        adj.append({"kind": "reclassify", "amount": _money(mm.group(1)),
                    "counterparty": mm.group(2).strip(),
                    "to_category": _map_line_item(mm.group(3)),
                    "reason": mm.group(3).strip()})

    # FX: foreign invoice settled in USD
    for mm in re.finditer(
        r"([\d,]+\.\d{2})\s*(EUR|GBP|KZT|CNY)[^.]{0,120}?" + MONEY, flat
    ):
        foreign, cur, usd = _money(mm.group(1)), mm.group(2), _money(mm.group(3))
        if foreign > 0:
            adj.append({"kind": "fx", "currency": cur, "rate_to_usd": usd / foreign,
                        "reason": "derived from settled invoice"})

    # amounts missing from the ledger export, restated in a document
    for mm in re.finditer(
        r"(TXN-[A-Z0-9\-]+)[^.]{0,160}?сумма не отражена в выгрузке[^.]{0,80}?"
        r"фактическая сумма операции составляет " + MONEY, flat):
        adj.append({"kind": "set_amount", "txn_id": mm.group(1),
                    "amount": _money(mm.group(2)), "reason": "restated from document"})

    # liabilities disclosed but never booked as a transaction
    for mm in re.finditer(
        r"обязательство по программе выходных пособий в размере " + MONEY, flat):
        adj.append({"kind": "disclosure", "name": "severance_liability",
                    "amount": _money(mm.group(1)), "reason": "disclosed severance programme"})

    # explicitly REJECTED reclassification -> must NOT be applied
    for mm in re.finditer(
        r"(TXN-[A-Z0-9\-]+)[^.]{0,300}?(первоначальная классификация[^.]{0,60}сохраняется|"
        r"корректировка для целей ковенантов не (?:производилась|требуется))", flat):
        adj.append({"kind": "rejected", "txn_id": mm.group(1),
                    "reason": "auditor considered and declined the reclassification"})

    # one-off add-backs approved by the auditor
    for mm in re.finditer(
        r"(TXN-[A-Z0-9\-]+)[^.]{0,200}?(обратн\w+ добавл\w+|подлежит обратному добавлению)", flat
    ):
        adj.append({"kind": "addback", "txn_id": mm.group(1), "reason": "auditor add-back"})

    floor = 0.0
    mf = re.search(r"признаются статьи в сумме не менее " + MONEY, flat)
    if mf:
        floor = _money(mf.group(1))
    if "разовые статьи" in flat.lower() or "Разовыми для целей ковенантов" in flat:
        for mm in re.finditer(r"«([^»]{3,60})»\s*:?\s*" + MONEY, flat):
            amount = _money(mm.group(2))
            if amount >= floor:
                adj.append({"kind": "addback", "amount": amount,
                            "counterparty": mm.group(1),
                            "reason": f"one-off item, above materiality {floor:,.2f}"})

    return {"is_binding": binding, "defers_to_report": defers, "adjustments": adj}


LINE_ITEM_MAP = {
    "процентные расходы": "interest_expense",
    "капитальные затраты": "capex",
    "операционные расходы": "opex",
    "коммунальные услуги": "utilities",
    "выручка": "revenue",
    "арендные платежи": "lease",
    "расходы на оплату труда": "payroll",
    "консультационные услуги": "professional_fees",
    "страхов": "insurance",
    "налог": "tax",
    "коммунальн": "utilities",
}


def _map_line_item(ru: str) -> str | None:
    key = ru.strip().lower().rstrip(".")
    for k, v in LINE_ITEM_MAP.items():
        if k in key:
            return v
    return None


# ------------------------------------------- amounts that must be derived ---
NBV_OPEN = re.compile(r"beginning of the year\s*\$?([\d,]+\.\d{2})", re.I)
NBV_CLOSE = re.compile(r"end of the year\s*\$?([\d,]+\.\d{2})", re.I)
DEPR = re.compile(r"Depreciation charge for the year\s*\$?([\d,]+\.\d{2})", re.I)


def group_capex(consolidated_text: str) -> float | None:
    """Group capital expenditure is NOT stated in the consolidated accounts.

    It has to be reconstructed from the property, plant and equipment
    roll-forward: additions = closing NBV - opening NBV + depreciation, given
    the note's statement that there were no disposals in the year.
    """
    flat = re.sub(r"\s+", " ", consolidated_text)
    o, c, d = NBV_OPEN.search(flat), NBV_CLOSE.search(flat), DEPR.search(flat)
    if not (o and c and d):
        return None
    return _money(c.group(1)) - _money(o.group(1)) + _money(d.group(1))


PLEDGE_RULE = re.compile(r"доля активов в залоге ниже\s*(\d{1,3}(?:\.\d+)?)\s*%", re.I)


def unrestricted_subsidiaries(kyc_text: str) -> list[str]:
    """Second KYC table: subsidiaries whose pledged-asset share falls below the
    stated floor sit outside the security perimeter and count as Unrestricted."""
    flat = re.sub(r"[ \t]+", " ", kyc_text)
    m = PLEDGE_RULE.search(re.sub(r"\s+", " ", flat))
    if not m:
        return []
    floor = float(m.group(1))
    start = flat.lower().find("обеспечительное покрытие")
    if start < 0:
        return []
    out = []
    for mm in re.finditer(r"([A-Z\u0410-\u042f][^\n:]{3,60}?)[:\s]{2,}(\d{1,3}(?:\.\d+)?)\s*%",
                          flat[start:]):
        if float(mm.group(2)) < floor:
            out.append(mm.group(1).strip(" .·|"))
    return out
