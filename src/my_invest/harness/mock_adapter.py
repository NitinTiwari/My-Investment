"""
Mock Environment and Deterministic Fixtures for My-Invest Agentic Harness.

Provides mock adapters for:
- AMFI Mutual Fund API (`api.mfapi.in`)
- Yahoo Finance ETF Lookups (`yfinance`)
- Web search calls (`DuckDuckGo`)
- Offline Simulated Agent Execution (for zero-cost, deterministic CI/CD runs)
"""
from contextlib import contextmanager
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import json

from my_invest.advisor_service import AdvisoryResult
from my_invest.tools.bank_fd_tool import get_indian_bank_fd_rates
from my_invest.tools.calculator_tool import (
    calculate_fd_maturity,
    calculate_mf_returns,
    calculate_portfolio_split,
)
from my_invest.tools.mutual_fund_tool import (
    get_mutual_fund_nav,
    get_top_indian_mutual_funds_by_category,
    search_indian_mutual_funds,
)

# Canned AMFI API responses
MOCK_AMFI_SEARCH_RESULTS = [
    {"schemeCode": 122639, "schemeName": "Parag Parikh Flexi Cap Fund - Direct Plan - Growth"},
    {"schemeCode": 118989, "schemeName": "HDFC Flexi Cap Fund - Direct Plan - Growth"},
    {"schemeCode": 120716, "schemeName": "UTI Nifty 50 Index Fund - Direct Plan - Growth Option"},
    {"schemeCode": 118778, "schemeName": "Nippon India Small Cap Fund - Direct Plan - Growth"},
]

MOCK_AMFI_NAV_RESULTS = {
    122639: {
        "meta": {
            "fund_house": "PPFAS Mutual Fund",
            "scheme_name": "Parag Parikh Flexi Cap Fund - Direct Plan - Growth",
            "scheme_category": "Equity Scheme - Flexi Cap Fund",
            "scheme_code": 122639,
        },
        "data": [{"date": "28-10-2026", "nav": "82.45"}],
    },
    118989: {
        "meta": {
            "fund_house": "HDFC Mutual Fund",
            "scheme_name": "HDFC Flexi Cap Fund - Direct Plan - Growth",
            "scheme_category": "Equity Scheme - Flexi Cap Fund",
            "scheme_code": 118989,
        },
        "data": [{"date": "28-10-2026", "nav": "1845.20"}],
    },
    120716: {
        "meta": {
            "fund_house": "UTI Mutual Fund",
            "scheme_name": "UTI Nifty 50 Index Fund - Direct Plan - Growth Option",
            "scheme_category": "Other Scheme - Index Funds",
            "scheme_code": 120716,
        },
        "data": [{"date": "28-10-2026", "nav": "178.90"}],
    },
}


def mock_requests_get(url: str, *args, **kwargs):
    """Mock implementation of requests.get for external financial endpoints."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    if "api.mfapi.in/mf/search" in url:
        mock_resp.json.return_value = MOCK_AMFI_SEARCH_RESULTS
        return mock_resp

    if "api.mfapi.in/mf/" in url:
        # Extract scheme code
        try:
            parts = url.split("api.mfapi.in/mf/")[1].split("/")
            code = int(parts[0])
            if code in MOCK_AMFI_NAV_RESULTS:
                mock_resp.json.return_value = MOCK_AMFI_NAV_RESULTS[code]
                return mock_resp
        except Exception:
            pass
        # Default fallback NAV
        mock_resp.json.return_value = {
            "meta": {
                "fund_house": "Standard AMC",
                "scheme_name": "Top Indian Equity Fund - Direct Growth",
                "scheme_category": "Equity Scheme - Multi Cap",
                "scheme_code": 122639,
            },
            "data": [{"date": "28-10-2026", "nav": "100.00"}],
        }
        return mock_resp

    mock_resp.json.return_value = {}
    return mock_resp


def mock_yfinance_ticker(ticker: str, *args, **kwargs):
    """Mock yfinance.Ticker to return realistic ETF quotes."""
    mock_t = MagicMock()
    mock_t.fast_info = {
        "last_price": 272.50 if "NIFTY" in ticker else 64.20,
        "currency": "INR",
        "day_high": 274.00,
        "day_low": 271.00,
        "year_high": 280.00,
        "year_low": 210.00,
    }
    return mock_t


@contextmanager
def mock_harness_environment():
    """
    Context manager that patches external HTTP network calls (AMFI API, Yahoo Finance)
    to enable deterministic, high-speed, zero-cost harness execution.
    """
    with patch("requests.get", side_effect=mock_requests_get), \
         patch("yfinance.Ticker", side_effect=mock_yfinance_ticker):
        yield


class SimulatedAdvisorRunner:
    """
    Simulates high-fidelity agent execution offline.
    Invokes real tools (FD rates, AMFI data, Calculators) and applies real guardrails
    to generate realistic, compliant advisory outputs without burning LLM API tokens.
    """

    def __init__(self, service: Any) -> None:
        self.service = service

    def run_advisory(self, prompt: str) -> AdvisoryResult:
        # Step 1: Run real input guardrail
        guard_res = self.service._input_guardrail.validate(prompt)
        if not guard_res.is_valid:
            return AdvisoryResult(
                prompt=prompt,
                text=guard_res.message,
                duration_ms=5.0,
                model_name="My-Invest Guardrail (Simulated)",
                llm_call_count=0,
                tool_call_count=0,
                tools_used=(),
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                is_cached=False,
                similarity_score=0.0,
                is_guarded=True,
            )

        prompt_lower = prompt.lower()
        tools_called: List[str] = []
        text_parts: List[str] = []

        is_fd = "fd" in prompt_lower or "fixed deposit" in prompt_lower
        is_mf = "mutual fund" in prompt_lower or "mf" in prompt_lower or "flexi" in prompt_lower
        is_etf = "etf" in prompt_lower

        if not is_fd and not is_mf and not is_etf:
            # Default to balanced
            is_fd = True
            is_mf = True

        # Simulate FD tools execution
        if is_fd:
            fd_rates = get_indian_bank_fd_rates("1 year")
            tools_called.append("get_indian_bank_fd_rates")
            maturity = calculate_fd_maturity(100000.0, 7.5, 1.0, 4)
            tools_called.append("calculate_fd_maturity")
            text_parts.append("### Part A: Bank Fixed Deposit (FD) Strategy")
            text_parts.append(fd_rates)
            text_parts.append(maturity)
            text_parts.append("All deposits are insured up to ₹5,00,000 per depositor per bank by DICGC.")

        # Simulate MF tools execution
        if is_mf:
            mf_list = get_top_indian_mutual_funds_by_category("flexi cap")
            tools_called.append("get_top_indian_mutual_funds_by_category")
            with mock_harness_environment():
                nav_info = get_mutual_fund_nav(122639)
            tools_called.append("get_mutual_fund_nav")
            returns = calculate_mf_returns(100000.0, 12.0, 3.0)
            tools_called.append("calculate_mf_returns")
            text_parts.append("### Part B: Indian Mutual Fund Strategy")
            text_parts.append(mf_list)
            text_parts.append(nav_info)
            text_parts.append(returns)

        # Simulate ETF tool execution
        if is_etf:
            tools_called.append("get_indian_etf_price")
            text_parts.append("### ETF Benchmark Tracking")
            text_parts.append("- **NIFTYBEES.NS**: Live Price ₹272.50 INR")
            text_parts.append("- **GOLDBEES.NS**: Live Price ₹64.20 INR")

        # Portfolio split
        if is_fd and is_mf:
            split = calculate_portfolio_split(200000.0, 50, 50)
            tools_called.append("calculate_portfolio_split")
            text_parts.insert(0, f"### Executive Summary & Portfolio Allocation\n{split}\n")

        raw_output = "\n\n".join(text_parts)

        # Step 3: Run real output guardrail
        guarded_out = self.service._output_guardrail.enforce(raw_output, prompt=prompt)

        return AdvisoryResult(
            prompt=prompt,
            text=guarded_out.text,
            duration_ms=45.0,
            model_name="Simulated Harness Agent (Offline)",
            llm_call_count=1,
            tool_call_count=len(tools_called),
            tools_used=tuple(tools_called),
            input_tokens=150,
            output_tokens=350,
            total_tokens=500,
            estimated_cost_usd=0.0001,
            is_cached=False,
            is_guarded=False,
        )

