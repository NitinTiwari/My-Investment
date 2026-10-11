"""
Production-Ready Input Guardrail for My-Invest.

Validates incoming prompts to:
1. Prevent Prompt Injection & Jailbreaks.
2. Filter Out-of-Domain / Off-Topic queries (e.g. 'what is time now?', trivia, code generation).
3. Ensure user queries fall within the domain of Indian Wealth Management, FDs, and Mutual Funds.
"""
from dataclasses import dataclass
import re
from typing import Optional


@dataclass(frozen=True)
class InputGuardrailResult:
    is_valid: bool
    reason: str
    message: str


# Common Prompt Injection & Jailbreak patterns
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"system\s+prompt",
    r"reveal\s+(your\s+)?(instructions|prompt)",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"jailbreak",
    r"\bdan\s+mode\b",
    r"bypass\s+(safety|content)\s+filters",
]

# Explicit off-topic categories (when no financial intent is present)
OFF_TOPIC_PATTERNS = [
    r"\b(what('?s|\s+is)?\s+(the\s+)?time(\s+now)?)\b",
    r"\b(current\s+time|what\s+time\s+is\s+it)\b",
    r"\b(what('?s|\s+is)?\s+(the\s+)?date(\s+today)?)\b",
    r"\b(what('?s|\s+is)?\s+(the\s+)?weather|temperature|is\s+it\s+raining)\b",
    r"\b(who\s+won\s+the\s+(match|game|world\s+cup|election))\b",
    r"\b(cricket\s+score|football\s+score)\b",
    r"\b(tell\s+me\s+a\s+joke|make\s+me\s+laugh|sing\s+a\s+song)\b",
    r"\b(recipe\s+for|how\s+to\s+cook|how\s+to\s+bake)\b",
    r"\b(write\s+(a\s+)?python\s+code|write\s+(a\s+)?javascript|write\s+(a\s+)?code\s+for)\b",
    r"\b(translate\s+(this|to)\s+[a-z]+)\b",
    r"\b(who\s+is\s+(the\s+)?(prime\s+minister|president|actor|actress))\b",
]

# Single-word financial terms (checked strictly against tokenized word set)
FINANCIAL_WORDS = {
    # Asset classes & instruments
    "fd", "fds", "mf", "mfs", "sip", "swp", "stp", "nav", "etf", "etfs",
    "equity", "debt", "bond", "bonds", "stock", "stocks", "shares", "gold",
    "sgb", "ppf", "epf", "nps", "elss", "cagr", "xirr",
    # Financial concepts & metrics
    "invest", "investing", "investment", "investments", "investor", "investors",
    "save", "saving", "savings", "wealth", "corpus", "capital", "portfolio",
    "portfolios", "allocation", "diversify", "diversification", "interest",
    "return", "returns", "compounding", "compound", "maturity", "yield",
    "inflation", "risk", "conservative", "aggressive", "tax", "taxation",
    "80c", "ltcg", "stcg", "dividend", "dividends", "retirement", "pension",
    # Currency & amounts
    "inr", "rs", "rupee", "rupees", "lakh", "lakhs", "lac", "lacs",
    "crore", "crores", "cr",
    # Banks & Financial entities
    "bank", "banks", "sbi", "hdfc", "icici", "pnb", "axis", "kotak",
    "canara", "equitas", "amfi", "sebi", "rbi", "dicgc", "uti", "nippon",
    "quant", "mirae", "tata",
}

# Multi-word financial phrases
FINANCIAL_PHRASES = [
    r"fixed\s+deposits?",
    r"term\s+deposits?",
    r"mutual\s+funds?",
    r"index\s+funds?",
    r"liquid\s+funds?",
    r"hybrid\s+funds?",
    r"balanced\s+advantage",
    r"overnight\s+funds?",
    r"arbitrage\s+funds?",
    r"gilt\s+funds?",
    r"large\s+cap",
    r"mid\s+cap",
    r"small\s+cap",
    r"flexi\s+cap",
    r"multi\s+cap",
    r"interest\s+rates?",
    r"risk\s+tolerance",
    r"risk\s+profile",
    r"tax\s+saving",
    r"emergency\s+funds?",
    r"wealth\s+creation",
    r"bank\s+of\s+baroda",
    r"union\s+bank",
    r"au\s+small\s+finance",
    r"parag\s+parikh",
]


def check_domain_relevance(prompt: str) -> tuple[bool, str]:
    """
    Validates whether a user prompt relates to personal finance and investments in India.
    Returns (is_valid: bool, reason_message: str).
    """
    cleaned = prompt.strip()
    if not cleaned:
        return False, "Investment prompt cannot be empty."

    # 1. Check explicit off-topic patterns
    for p in OFF_TOPIC_PATTERNS:
        if re.search(p, cleaned, re.IGNORECASE):
            return False, "Off-topic query detected (e.g. weather, time, trivia, or coding)."

    # 2. Domain relevance check
    words = set(re.findall(r"\b[a-zA-Z0-9₹]+\b", cleaned.lower()))
    prompt_lower = cleaned.lower()

    # Check for currency symbols or explicit amounts (e.g. ₹, Rs 50000, 2 Lakhs, 50k)
    has_currency_amount = bool(
        re.search(r"[₹]|(\brs\.?\s*\d+)|(\d+\s*(lakh|lac|crore|cr|k|thousand|inr))", prompt_lower)
        or any(w in words for w in ["inr", "lakh", "lakhs", "lac", "lacs", "crore", "crores", "cr"])
    )

    # Check exact whole-word financial keyword match
    has_financial_word = any(w in words for w in FINANCIAL_WORDS)

    # Check multi-word financial phrases match
    compiled_phrases = [re.compile(r"\b" + p + r"\b", re.IGNORECASE) for p in FINANCIAL_PHRASES]
    has_financial_phrase = any(cp.search(cleaned) for cp in compiled_phrases)

    if not (has_financial_word or has_financial_phrase or has_currency_amount):
        return False, "Query does not appear related to Fixed Deposits, Mutual Funds, or financial planning."

    return True, ""


from my_invest.guardrails.llm_judge import LLMJudge, LLMJudgeVerdict


class InputGuardrail:
    """
    Validates user prompts using a lightweight LLM-as-a-Judge (Qwen / Llama Prompt Guard)
    with a fast rule-based fallback safety net.
    """

    def __init__(
        self,
        use_llm_judge: bool = True,
        llm_judge: Optional[LLMJudge] = None,
    ) -> None:
        self.use_llm_judge = use_llm_judge
        self._llm_judge = llm_judge if llm_judge is not None else (LLMJudge() if use_llm_judge else None)
        self._compiled_injection = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
        self._compiled_off_topic = [re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS]
        self._compiled_phrases = [re.compile(r"\b" + p + r"\b", re.IGNORECASE) for p in FINANCIAL_PHRASES]

    def _build_redirect_message(self, specific_reason: Optional[str] = None) -> str:
        reason_line = f"\n\n🔍 **Evaluation**: *{specific_reason}*" if specific_reason else ""
        return (
            "👋 **Hello! I am My-Invest, your dedicated Indian Wealth & Investment Advisor.**\n\n"
            f"I specialize exclusively in **Bank Fixed Deposits (FDs)**, **Mutual Funds**, **ETFs**, "
            f"and **portfolio allocation strategies**.{reason_line}\n\n"
            "💡 **Try asking me an investment query, for example:**\n"
            "- *\"I want to invest ₹2 Lakhs for 3 years with balanced risk.\"*\n"
            "- *\"What are the highest bank FD rates in India right now?\"*\n"
            "- *\"Recommend top Flexi Cap and Large Cap mutual funds for a 5-year goal.\"*\n"
            "- *\"How should I split ₹5 Lakhs between SBI FD and Parag Parikh Flexi Cap?\"*"
        )

    def validate(self, prompt: str) -> InputGuardrailResult:
        """
        Validates the user prompt using LLM-as-a-Judge / Prompt Guard.
        Falls back to rule-based analysis if the model is unreachable.
        """
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            return InputGuardrailResult(
                is_valid=False,
                reason="empty_prompt",
                message="⚠️ **Investment prompt cannot be empty.** Please enter your investment goal or amount.",
            )

        # 1. Fast security pre-filter for direct prompt injection / jailbreak attempts
        for pattern in self._compiled_injection:
            if pattern.search(cleaned_prompt):
                return InputGuardrailResult(
                    is_valid=False,
                    reason="prompt_injection",
                    message=(
                        "🛡️ **Security Guardrail Triggered**: This query contains system-override or injection patterns. "
                        "I am configured strictly as an Indian Wealth and Investment Advisor. "
                        "Please provide an investment planning query."
                    ),
                )

        # 2. Primary: Lightweight LLM-as-a-Judge / Prompt Guard evaluation
        if self.use_llm_judge and self._llm_judge:
            verdict = self._llm_judge.evaluate(cleaned_prompt)
            if verdict is not None:
                if verdict.allowed:
                    return InputGuardrailResult(
                        is_valid=True,
                        reason="llm_judge_allowed",
                        message="",
                    )
                else:
                    if getattr(verdict, "is_injection", False):
                        return InputGuardrailResult(
                            is_valid=False,
                            reason="prompt_injection",
                            message=(
                                f"🛡️ **Security Guardrail Triggered**: Prompt injection or jailbreak detected by {verdict.model_name or 'Prompt Guard'}. "
                                "I am configured strictly as an Indian Wealth and Investment Advisor. "
                                "Please provide an investment planning query."
                            ),
                        )
                    return InputGuardrailResult(
                        is_valid=False,
                        reason="llm_judge_rejected",
                        message=self._build_redirect_message(specific_reason=verdict.reason),
                    )

        # 3. Fallback: Fast Rule-Based Domain Safety Net (if LLM Judge is offline or disabled)
        is_valid, domain_reason = check_domain_relevance(cleaned_prompt)
        if not is_valid:
            if any(p.search(cleaned_prompt) for p in self._compiled_off_topic):
                return InputGuardrailResult(
                    is_valid=False,
                    reason="off_topic",
                    message=self._build_redirect_message(specific_reason="Off-topic query detected by safety net."),
                )
            return InputGuardrailResult(
                is_valid=False,
                reason="out_of_domain",
                message=(
                    "👋 **I can only assist with Indian investments and wealth advisory.**\n\n"
                    "Your question does not appear to be related to Fixed Deposits, Mutual Funds, or financial planning.\n\n"
                    "💡 **Ask an investment question, for example:**\n"
                    "- *\"Plan a portfolio of ₹3,00,000 for 5 years with moderate risk.\"*\n"
                    "- *\"Which bank gives the best FD interest rate for senior citizens?\"*\n"
                    "- *\"Compare FD vs Mutual Fund returns for a 3-year investment.\"*"
                ),
            )

        return InputGuardrailResult(
            is_valid=True,
            reason="valid_financial_query",
            message="",
        )
