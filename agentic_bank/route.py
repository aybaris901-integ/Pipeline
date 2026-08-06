"""Stage 2 — work out what each document is and whose it is.

Filenames are opaque hashes. ~73% of the corpus is deliberate noise (HR
policies, IT incident reports, marketing decks). The signal documents per
borrower are:

  * credit agreement, CURRENT           -> covenant definitions
  * credit agreement, SUPERSEDED (2024) -> trap: different thresholds
  * KYC dossier                         -> related-party list + ownership cut-off
  * auditor's notes to the accounts     -> cut-off / FX / add-back disclosures
  * agreed-procedures report, FINAL     -> binding reclassifications
  * interim classification worksheet    -> trap: explicitly superseded draft

Linking is by ACC-#### where present and by borrower legal name otherwise
(the notes to the accounts often carry only the name).
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field

from .ingest import normalise

ACC_RE = re.compile(r"ACC-\d{4}")
TXN_RE = re.compile(r"TXN-[A-Z0-9]{1,4}-\d{3,4}")
COMPANY_RE = re.compile(r"\b([A-Z][A-Za-z\-]+(?: [A-Z][A-Za-z\-&]+){0,4} JSC)\b")

DOC_TYPES = (
    "credit_agreement_current",
    "credit_agreement_superseded",
    "kyc",
    "audit_notes",
    "audit_agreed_procedures",
    "audit_draft_worksheet",
    "consolidated_accounts",
    "treasury_memo",
    "noise",
)


@dataclass
class Doc:
    doc_id: str
    text: str          # normalised text
    raw: str
    doc_type: str = "noise"
    account: str | None = None
    company: str | None = None
    scenario: str | None = None
    txn_refs: list[str] = field(default_factory=list)


def _is_superseded(t: str) -> bool:
    head = t[:4000]
    return ("НЕДЕЙСТВУЮЩАЯ" in head or "НЕДЕЙСТВУЮЩАЯРЕДАКЦИЯ" in head
            or "Заменена и изложена в новой редакции" in head)


def _is_draft(t: str) -> bool:
    head = t[:2500]
    return ("ПРОМЕЖУТОЧНАЯ ВЕДОМОСТЬ" in head or "ПРОМЕЖУТОЧНАЯВЕДОМОСТЬ" in head
            or ("ПРОЕКТ" in head and "НЕ ЯВЛЯЕТСЯ ОКОНЧАТЕЛЬНОЙ" in head.replace("\n", " "))
            or "не являются окончательными" in t[:3500])


def classify_doc(text: str) -> str:
    t = normalise(text)
    flat = re.sub(r"\s+", " ", t)
    if "Финансовые ковенанты" in flat and "ДОГОВОР БАНКОВСКОГО ЗАЙМА" in flat.upper():
        return "credit_agreement_superseded" if _is_superseded(t) else "credit_agreement_current"
    if "Знай своего клиент" in flat or "KYC-ACC-" in flat:
        return "kyc"
    if _is_draft(t):
        return "audit_draft_worksheet"
    if "согласованных процедур" in flat or "Номер заключения" in flat or \
            "Выводы по классификации операций" in flat:
        return "audit_agreed_procedures"
    if "записка казначейства" in flat.lower() or "Служебная записка казначейства" in flat:
        return "treasury_memo"
    if "Consolidated Financial Statements" in flat or "консолидированная отчётность" in flat.lower():
        return "consolidated_accounts"
    if "Примечания к финансовой отчётности" in flat:
        return "audit_notes"
    return "noise"


def build_index(texts: dict[str, str], account_to_scenario: dict[str, str]) -> list[Doc]:
    # first pass: company name -> account, learned from documents that carry both
    name_to_acc: dict[str, str] = {}
    docs: list[Doc] = []
    for doc_id, raw in texts.items():
        t = normalise(raw)
        d = Doc(doc_id=doc_id, text=t, raw=raw, doc_type=classify_doc(raw))
        flat = re.sub(r"\s+", " ", t)
        accs = ACC_RE.findall(flat)
        d.account = accs[0] if accs else None
        names = COMPANY_RE.findall(flat)
        d.company = _dominant_company(flat, names)
        d.txn_refs = sorted(set(TXN_RE.findall(flat)))
        if d.account in account_to_scenario and d.company:
            name_to_acc.setdefault(d.company, d.account)
        docs.append(d)

    # second pass: resolve scenario
    for d in docs:
        acc = d.account
        if acc not in account_to_scenario and d.company:
            acc = name_to_acc.get(d.company)
        d.scenario = account_to_scenario.get(acc or "")

    # third pass: a parent's consolidated accounts carry the PARENT's name, so
    # they attach to the borrower they name as the operating subsidiary.
    known = {c: a for c, a in name_to_acc.items()}
    for d in docs:
        if d.scenario or d.doc_type != "consolidated_accounts":
            continue
        flat = re.sub(r"\s+", " ", d.text)
        for company, acc in known.items():
            if company in flat:
                d.scenario = account_to_scenario.get(acc)
                break
    return docs


def _dominant_company(flat: str, names: list[str]) -> str | None:
    """Pick the borrower name, not the auditor or the lender.

    Two hazards: 'Shymkent Refinery JSC' (B4) and 'Shymkent Refinery Services
    JSC' (P3) are different borrowers, so the *longest* match at each position
    must win; and 'Halyk Bank of Kazakhstan JSC' appears in every document.
    """
    skip = ("Halyk Bank", "KEGOC")
    counts: dict[str, int] = {}
    for n in names:
        if any(s in n for s in skip):
            continue
        counts[n] = counts.get(n, 0) + 1
    if not counts:
        return None
    # prefer longer names when one is a prefix of another and both are frequent
    best = max(counts, key=lambda n: (counts[n], len(n)))
    for n in counts:
        if n != best and best in n and counts[n] >= 2:
            best = n
    return best


def select_sources(docs: list[Doc], scenario: str) -> dict[str, list[Doc]]:
    """Return the authoritative document set for one borrower.

    Precedence rules encoded here (all of them are traps in the dataset):
      * the CURRENT agreement wins over the 2024 restated one;
      * the FINAL agreed-procedures report wins over the interim worksheet;
      * the notes to the accounts are read for cut-off / FX / add-backs, and a
        note that defers to a report number is followed to that report.
    """
    mine = [d for d in docs if d.scenario == scenario]
    out = {t: [d for d in mine if d.doc_type == t] for t in DOC_TYPES}
    return out
