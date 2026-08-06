"""Global configuration and the fixed category taxonomy."""
from __future__ import annotations
import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------- taxonomy ---
# A *closed* list. Every ledger line is mapped to exactly one of these.
# Keeping it closed is what makes covenant maths deterministic downstream.
CATEGORIES = [
    "revenue",            # core trading inflows: "... sales settlement"
    "opex",               # core operating/maintenance: "... servicing and operating costs"
    "capex",              # "Purchase of ... equipment", capitalised works
    "lease",              # rent and lease payments
    "payroll",            # staff costs
    "utilities",          # electricity / water / gas / heating
    "interest_expense",   # interest paid
    "interest_income",    # interest received
    "tax",                # taxes and duties paid
    "insurance",          # insurance premiums
    "marketing",          # advertising / marketing
    "telecom",            # telecom and connectivity
    "professional_fees",  # advisory / consulting / management retainers
    "financing_inflow",   # loan drawdowns, facility proceeds
    "credit_refund",      # money coming *back*: refunds, rebates, deposits released
    "other",
]

EXPENSE_CATEGORIES = {
    "opex", "capex", "lease", "payroll", "utilities", "interest_expense",
    "tax", "insurance", "marketing", "telecom", "professional_fees",
}
INFLOW_CATEGORIES = {"revenue", "interest_income", "financing_inflow", "credit_refund"}


# ------------------------------------------------------------------- paths ---
@dataclass
class Paths:
    dataset: str
    work: str = ".cache"

    @property
    def ledger(self) -> str:
        return os.path.join(self.dataset, "master_ledger_2025.csv")

    @property
    def documents(self) -> str:
        return os.path.join(self.dataset, "documents")

    @property
    def template(self) -> str:
        return os.path.join(self.dataset, "submission_template.json")

    @property
    def ground_truth(self) -> str:
        return os.path.join(self.dataset, "ground_truth.json")

    @property
    def text_cache(self) -> str:
        return os.path.join(self.work, "text")

    @property
    def llm_cache(self) -> str:
        return os.path.join(self.work, "llm")


# --------------------------------------------------------------------- llm ---
@dataclass
class LLMConfig:
    provider: str = "gemini"
    model: str = os.environ.get("AB_MODEL", "gemini-2.5-flash")
    api_key: str = os.environ.get("GEMINI_API_KEY", "")
    base_url: str = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
    max_tokens: int = 8000
    temperature: float = 0.0
    max_retries: int = 4
    # odd vote count for majority voting on the noisy transaction-classification stage
    self_consistency: int = int(os.environ.get("AB_VOTES", "1"))


@dataclass
class Submission:
    team: str = os.environ.get("AB_TEAM", "your-team-name")
    contact_email: str = os.environ.get("AB_EMAIL", "you@example.com")
    model: str = field(default_factory=lambda: os.environ.get("AB_MODEL", "gemini-2.5-flash"))
