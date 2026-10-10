import pytest
from my_invest.tools.bank_fd_tool import BENCHMARK_FD_RATES, get_indian_bank_fd_rates
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


def test_benchmark_fd_rates_structure():
    assert "Public Sector Banks" in BENCHMARK_FD_RATES
    assert "Leading Private Banks" in BENCHMARK_FD_RATES
    assert len(BENCHMARK_FD_RATES["Public Sector Banks"]) > 0


def test_get_indian_bank_fd_rates():
    rates = get_indian_bank_fd_rates("1 year")
    assert "State Bank of India (SBI)" in rates
    assert "HDFC Bank" in rates
    assert "DICGC" in rates


def test_calculate_fd_maturity():
    # ₹1,00,000 at 7% for 1 year compounded quarterly
    result = calculate_fd_maturity(100000, 7.0, 1.0, 4)
    assert "Principal Deposited" in result
    assert "Maturity Value" in result
    assert "107,185" in result or "107,186" in result  # 100000 * (1 + 0.07/4)^4 ~= 107185.9


def test_calculate_mf_returns():
    # ₹1,00,000 at 12% CAGR for 3 years
    result = calculate_mf_returns(100000, 12.0, 3.0)
    assert "Projected Future Value" in result
    assert "140,492" in result  # 100000 * 1.12^3 = 140492.8


def test_calculate_portfolio_split():
    result = calculate_portfolio_split(200000, 50, 50)
    assert "100,000.00" in result


def test_search_indian_mutual_funds():
    res = search_indian_mutual_funds("Parag Parikh")
    assert "Scheme Code" in res or "Parag Parikh" in res


def test_get_mutual_fund_nav():
    # Scheme 122639 is Parag Parikh Flexi Cap Direct Growth
    nav_res = get_mutual_fund_nav(122639)
    assert "PPFAS" in nav_res or "Parag Parikh" in nav_res or "NAV" in nav_res


def test_get_top_indian_mutual_funds_by_category():
    res = get_top_indian_mutual_funds_by_category("flexi cap")
    assert "Flexi Cap" in res
    assert "Parag Parikh" in res


def test_advisory_result_telemetry():
    from my_invest.advisor_service import AdvisoryResult
    result = AdvisoryResult(
        prompt="Test investment query",
        text="Sample output strategy",
        duration_ms=1250.0,
        model_name="qwen/qwen3.8-27b",
        llm_call_count=2,
        tool_call_count=3,
        tools_used=("get_indian_bank_fd_rates", "calculate_fd_maturity"),
        input_tokens=450,
        output_tokens=350,
        total_tokens=800,
    )
    assert result.llm_call_count == 2
    assert result.tool_call_count == 3
    assert "get_indian_bank_fd_rates" in result.tools_used
    assert result.total_tokens == 800

