"""Offline covenant parser.

Not a replacement for the model — a cross-check. It recognises the archetypes
by the phrases the agreements actually use, so when the model's spec disagrees
with this one the run stops and shows both. Anything it cannot recognise is
returned with `unparsed=True` rather than guessed at.
"""
from __future__ import annotations
import re

MONEY = re.compile(r"\$\s?([\d,]+(?:\.\d{2})?)")
MULT = re.compile(r"(\d+(?:\.\d+)?)\s?x")

EBITDA = {"categories": ["revenue", "opex"]}


def _num(s: str) -> float:
    return float(s.replace(",", "").replace("$", "").strip())


def split_clauses(section: str) -> dict[str, str]:
    flat = re.sub(r"\s+", " ", section)
    parts: dict[str, str] = {}
    marks = [(m.start(), m.group(1)) for m in re.finditer(r"Пункт (\d+\.\d+)", flat)]
    for i, (pos, num) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(flat)
        parts[num] = flat[pos:end]
    return parts


def parse_clause(num: str, text: str, period=("2025-01-01", "2025-12-31")) -> dict:
    t = text
    low = t.lower()
    spec: dict = {"clause": num, "period_start": period[0], "period_end": period[1],
                  "title": t[:120], "notes": ""}

    monies = [_num(m) for m in MONEY.findall(t)]
    mults = [float(m) for m in MULT.findall(t)]

    def is_max() -> bool:
        return bool(re.search(r"не\s+(?:допускать|вправе|должн\w+)[^.]{0,80}?превы|"
                              r"превышал|превысил|не превышало|составили более|свыше", low))

    direction = "max" if is_max() else "min"

    # ---- ratio archetypes -------------------------------------------------
    if "капиталоём" in low or "capital intensity" in low:
        spec.update(kind="ratio", direction="max", threshold=mults[0],
                    numerator={"categories": ["capex"]},
                    denominator={"categories": ["opex", "lease"]})
    elif "покрытия процентов" in low:
        spec.update(kind="ratio", direction="min", threshold=mults[0],
                    numerator=EBITDA, denominator={"categories": ["interest_expense"]},
                    notes="EBITDA / interest expense")
    elif "cover of applications by sources" in low or "поступлений по финансированию" in low and "операционных и капитальных" in low:
        spec.update(kind="ratio", direction="min", threshold=mults[0],
                    numerator={"categories": ["revenue", "financing_inflow"]},
                    denominator={"categories": ["opex", "capex"]})
    elif "springing" in low or ("поступлений по финансированию к ebitda" in low):
        cond = {"component": {"categories": ["financing_inflow"]}, "operator": ">",
                "value": max(monies) if monies else 0.0}
        spec.update(kind="ratio", direction="max", threshold=mults[0],
                    numerator={"categories": ["financing_inflow"]}, denominator=EBITDA,
                    springing_condition=cond)
    elif "рентабельность по ebitda" in low or "скорректированной ebitda к выручке" in low:
        spec.update(kind="ratio", direction="min", threshold=mults[0],
                    numerator={"categories": ["revenue", "opex"], "include_addbacks": True},
                    denominator={"categories": ["revenue"]})
    elif "капитальных затрат группы к ebitda" in low:
        spec.update(kind="ratio", direction="max", threshold=mults[0],
                    numerator={"categories": [], "extras": ["group_capex"]},
                    denominator=EBITDA)
    elif "доля платежей связанным сторонам в операционных расходах" in low:
        spec.update(kind="ratio", direction="max", threshold=mults[0],
                    numerator={"categories": [c for c in _all_expense()],
                               "related_party_only": True},
                    denominator={"categories": ["opex"]})
    elif "покрытие расходов на персонал и коммунальные" in low:
        spec.update(kind="ratio", direction="min", threshold=mults[0],
                    numerator={"categories": ["revenue"]},
                    denominator={"categories": ["payroll", "utilities"]})
    elif "налоговой и коммунальной нагрузки" in low:
        spec.update(kind="ratio", direction="max", threshold=mults[0],
                    numerator={"categories": ["tax", "utilities"], "extras": ["accrued_tax"]},
                    denominator=EBITDA)
    elif "страховое покрытие расходов на содержание" in low or "страховых премий" in low:
        spec.update(kind="ratio", direction="min", threshold=mults[0],
                    numerator={"categories": ["insurance"]},
                    denominator={"categories": ["lease", "utilities"]})
    elif "активов, переданных неограниченным" in low:
        spec.update(kind="ratio", direction="max", threshold=mults[0],
                    numerator={"categories": ["capex"], "unrestricted_only": True},
                    denominator={"categories": ["capex"]})
    elif "related-party payments as a proportion of revenue" in low or (
            "связанным сторонам" in low or "аффилированн" in low) and mults:
        spec.update(kind="ratio", direction="max", threshold=mults[0],
                    numerator={"categories": _all_expense(), "related_party_only": True},
                    denominator={"categories": ["revenue"]})

    # ---- amount archetypes ------------------------------------------------
    elif "платежи связанным сторонам" in low or "связанным сторонам" in low or "аффилированн" in low:
        spec.update(kind="amount", direction="max", threshold=monies[0],
                    numerator={"categories": _all_expense(), "related_party_only": True})
    elif "overhead line ceiling" in low or "отдельная статья накладных" in low or (
            "по наибольшей из указанных сумм" in low):
        spec.update(kind="amount", direction="max", threshold=monies[0],
                    numerator={"categories": ["payroll", "utilities"], "largest_line_only": True})
    elif "выручка за вычетом наибольшей" in low:
        spec.update(kind="amount", direction="min", threshold=monies[0],
                    numerator={"categories": ["revenue"],
                               "subtract_largest_of": ["payroll", "tax"]})
    elif "выручку за четвёртый" in low or "выручка за четвёртый квартал" in low:
        spec.update(kind="amount", direction="min", threshold=monies[0],
                    numerator={"categories": ["revenue"], "quarter": 4})
    elif "обязательства по персоналу" in low:
        spec.update(kind="amount", direction="max", threshold=monies[0],
                    numerator={"categories": ["payroll"], "extras": ["severance_liability"]})
    elif "«выручка»" in low or "выручк" in low:
        spec.update(kind="amount", direction="min", threshold=monies[0],
                    numerator={"categories": ["revenue"]})
    elif "«капитальные затраты»" in low or "капитальные затраты" in low:
        spec.update(kind="amount", direction="max", threshold=monies[0],
                    numerator={"categories": ["capex"]})
    else:
        spec.update(kind="amount", direction=direction,
                    threshold=(monies[0] if monies else (mults[0] if mults else 0.0)),
                    numerator={"categories": []}, unparsed=True)
    return spec


def _all_expense() -> list[str]:
    """A related-party covenant counts the payment 'whatever line it sits on'."""
    from .config import EXPENSE_CATEGORIES
    return sorted(EXPENSE_CATEGORIES)


def extract_covenants(section_text: str) -> list[dict]:
    return [parse_clause(num, body) for num, body in sorted(split_clauses(section_text).items())]
