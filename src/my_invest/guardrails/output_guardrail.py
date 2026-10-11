"""
Production-Ready Output Guardrail for My-Invest.

Inspects model responses before delivery to ensure:
1. Regulatory Compliance (Mandatory SEBI Market Risk & DICGC Bank Insurance Disclaimers).
2. Elimination of Unrealistic or Illegal Guarantees on Market-Linked Assets.
3. Proper Markdown Formatting and Disclaimer Sanitization.
"""
from dataclasses import dataclass
import re

SEBI_DISCLAIMER_BLOCK = """

---

> ⚠️ **Regulatory & Market Risk Notice (SEBI Compliance)**:  
> **Mutual Fund investments are subject to market risks.** Please read all scheme-related documents carefully before investing. Historical NAV performance and past CAGR are indicative and do not guarantee future returns. Bank Fixed Deposits are protected up to ₹5,00,000 per depositor per bank by DICGC (RBI subsidiary). This blueprint is an AI-assisted research tool for informational purposes and does not constitute formal SEBI-registered investment advice.
"""


@dataclass(frozen=True)
class OutputGuardrailResult:
    text: str
    disclaimer_added: bool
    guarantee_sanitized: bool


class OutputGuardrail:
    """
    Enforces SEBI educational disclaimers and checks for safety violations
    in generated investment blueprints.
    """

    def enforce(self, text: str, prompt: str = "") -> OutputGuardrailResult:
        """
        Validates and enriches the generated advisory blueprint.
        Ensures compliance disclaimers and removes false return guarantees.
        """
        if not text:
            return OutputGuardrailResult(
                text=SEBI_DISCLAIMER_BLOCK.strip(),
                disclaimer_added=True,
                guarantee_sanitized=False,
            )

        sanitized_text = text
        guarantee_sanitized = False

        # 1. Sanitize false guarantee claims on market-linked investments
        # e.g., "guaranteed 15% return in mutual funds" -> "projected/historical 15% return in mutual funds"
        unsafe_guarantee_patterns = [
            (
                r"\bguaranteed\s+(\d+%\s*(returns?|cagr))\s+(in|from|on)\s+(mutual\s+funds?|equity|stocks?)",
                r"projected/historical \1 \3 \4",
            ),
            (
                r"\b100%\s*(risk-free|guaranteed)\s+(mutual\s+fund|equity|stock)",
                r"market-linked \2",
            ),
        ]

        for pattern, replacement in unsafe_guarantee_patterns:
            if re.search(pattern, sanitized_text, re.IGNORECASE):
                sanitized_text = re.sub(pattern, replacement, sanitized_text, flags=re.IGNORECASE)
                guarantee_sanitized = True

        # 2. Check if SEBI / market risk disclaimer is present
        has_sebi_notice = bool(
            re.search(r"subject\s+to\s+market\s+risks?", sanitized_text, re.IGNORECASE)
            or re.search(r"sebi\s+(registered|compliance|disclaimer|regulations?)", sanitized_text, re.IGNORECASE)
        )

        disclaimer_added = False
        if not has_sebi_notice:
            sanitized_text = sanitized_text.rstrip() + SEBI_DISCLAIMER_BLOCK
            disclaimer_added = True

        return OutputGuardrailResult(
            text=sanitized_text,
            disclaimer_added=disclaimer_added,
            guarantee_sanitized=guarantee_sanitized,
        )
