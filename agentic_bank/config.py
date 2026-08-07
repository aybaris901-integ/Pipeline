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
# Groq's OpenAI-compatible endpoint; each provider resolves its own API key
# and base URL from the environment (never hard-coded).
_DEFAULT_BASE_URL = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com",
}
_API_KEY_ENV = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY"}
# Fallback default *text* model per provider, used only when LLM_MODEL (or
# an explicit model= override) is absent — e.g. LLM_PROVIDER=gemini with no
# LLM_MODEL set should not inherit the Groq-shaped "llama-3.1-8b-instant"
# default, which isn't a Gemini model.
_DEFAULT_MODEL = {"groq": "llama-3.1-8b-instant", "gemini": "gemini-2.5-flash"}
# Same idea for the vision/OCR model.
_DEFAULT_VISION_MODEL = {"groq": "qwen/qwen3.6-27b", "gemini": "gemini-2.5-flash"}


def _base_url_for(provider: str) -> str:
    if provider not in _DEFAULT_BASE_URL:
        raise ValueError(f"unknown LLM provider {provider!r}; expected 'groq' or 'gemini'")
    env = "GROQ_BASE_URL" if provider == "groq" else "GEMINI_BASE_URL"
    return os.environ.get(env, _DEFAULT_BASE_URL[provider])


def _api_key_for(provider: str) -> str:
    if provider not in _API_KEY_ENV:
        raise ValueError(f"unknown LLM provider {provider!r}; expected 'groq' or 'gemini'")
    return os.environ.get(_API_KEY_ENV[provider], "")


@dataclass
class LLMConfig:
    # Required model chain (see module docstring / README): Groq's
    # llama-3.1-8b-instant handles routine/high-volume calls; failures that
    # look like a model-quality problem (bad JSON, failed schema
    # validation) or an exhausted rate limit escalate to the fallback, then
    # the strong fallback. All three are configuration, not hard-wired.
    provider: str = os.environ.get("LLM_PROVIDER", "groq")
    # Left blank, `model` and `vision_model` fall back to a provider-specific
    # default in __post_init__ — a bare Groq-shaped default would be wrong
    # for LLM_PROVIDER=gemini.
    model: str = os.environ.get("LLM_MODEL", "")
    fallback_model: str = os.environ.get("LLM_FALLBACK_MODEL", "openai/gpt-oss-20b")
    strong_fallback_model: str = os.environ.get("LLM_STRONG_FALLBACK_MODEL", "openai/gpt-oss-120b")

    # Vision/OCR (transcribing the handful of scanned, image-only PDF pages)
    # is configured independently since it needs a multimodal model — none
    # of the three text models above accept images. Left blank, it follows
    # `provider` (resolved in __post_init__, so an explicit `provider=`
    # override is honoured too, not just the env var).
    vision_provider: str = os.environ.get("LLM_VISION_PROVIDER", "")
    vision_model: str = os.environ.get("LLM_VISION_MODEL", "")

    api_key: str = ""
    base_url: str = ""
    vision_api_key: str = ""
    vision_base_url: str = ""

    max_tokens: int = 8000
    temperature: float = 0.0
    max_retries: int = 4
    # odd vote count for majority voting on the noisy transaction-classification stage
    self_consistency: int = int(os.environ.get("AB_VOTES", "1"))
    verbose: bool = False

    def __post_init__(self):
        if not self.vision_provider:
            self.vision_provider = self.provider
        if not self.model:
            self.model = _DEFAULT_MODEL.get(self.provider, "")
        if not self.vision_model:
            self.vision_model = _DEFAULT_VISION_MODEL.get(self.vision_provider, "")
        if not self.api_key:
            self.api_key = _api_key_for(self.provider)
        if not self.base_url:
            self.base_url = _base_url_for(self.provider)
        if not self.vision_api_key:
            self.vision_api_key = _api_key_for(self.vision_provider)
        if not self.vision_base_url:
            self.vision_base_url = _base_url_for(self.vision_provider)


@dataclass
class Submission:
    team: str = os.environ.get("AB_TEAM", "your-team-name")
    contact_email: str = os.environ.get("AB_EMAIL", "you@example.com")
    model: str = field(default_factory=lambda: os.environ.get(
        "AB_MODEL", os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")))
