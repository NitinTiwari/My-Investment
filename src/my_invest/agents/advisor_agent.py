from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.openrouter import OpenRouter
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

from my_invest.settings import settings
from my_invest.tools.bank_fd_tool import get_indian_bank_fd_rates
from my_invest.tools.calculator_tool import (
    calculate_fd_maturity,
    calculate_mf_returns,
    calculate_portfolio_split,
)
from my_invest.tools.mutual_fund_tool import (
    get_indian_etf_price,
    get_mutual_fund_nav,
    get_top_indian_mutual_funds_by_category,
    search_indian_mutual_funds,
)


def create_investment_advisor_agent(
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Agent:
    """
    Builds the flagship Investment Advisor Agent with tools for Bank FD rates,
    Indian Mutual Funds (AMFI live NAVs), ETF pricing, and financial math calculators.
    Uses OpenRouter LLM Gateway if configured, with direct Groq fallback.
    """
    temp = temperature if temperature is not None else settings.temperature
    tokens = max_tokens if max_tokens is not None else settings.max_tokens

    if settings.gateway_enabled:
        primary_model = settings.llm_gateway_model
        fallback_models = list(dict.fromkeys([
            primary_model,
            "qwen/qwen3.8-27b",
            "nvidia/nemotron-3-super-120b-a12b:free",
        ]))[:3]
        model = OpenRouter(
            id=primary_model,
            models=fallback_models,
            api_key=settings.llm_gateway_api_key,
            base_url=settings.llm_gateway_url,
            max_tokens=tokens,
            temperature=temp,
            default_headers={
                "HTTP-Referer": "https://github.com/my-invest",
                "X-Title": "My-Invest Financial Advisor",
            },
        )
    else:
        model = Groq(
            id=settings.model_name,
            api_key=settings.groq_api_key,
            max_tokens=tokens,
            temperature=temp,
        )

    advisor_agent = Agent(
        model=model,
        tool_call_limit=settings.agent_tool_call_limit,
        tools=[
            get_indian_bank_fd_rates,
            search_indian_mutual_funds,
            get_mutual_fund_nav,
            get_top_indian_mutual_funds_by_category,
            get_indian_etf_price,
            calculate_fd_maturity,
            calculate_mf_returns,
            calculate_portfolio_split,
            YFinanceTools(enable_stock_price=True),
            DuckDuckGoTools(),
        ],
        instructions=[
            "You are an expert Indian Wealth and Investment Advisor specializing in Indian Bank Fixed Deposits (FDs), Indian Mutual Funds (MFs), and Exchange Traded Funds (ETFs).",
            "When a user provides an investment prompt (for example: 'I want to invest 2 lacs inr in Indian bank fixed deposits and/or in Indian mutual funds'):",
            "",
            "1. PARSE INTENT & CORPUS:",
            "   - Extract the total amount (e.g. 2 Lacs = ₹2,00,000 INR).",
            "   - Determine the investment horizon (short term 1-2 yrs, medium 3-5 yrs, long term 5+ yrs) and risk tolerance (Conservative, Moderate, Aggressive). Default to Moderate/Balanced if unspecified.",
            "",
            "2. GATHER DATA USING TOOLS:",
            "   - Call `get_indian_bank_fd_rates` to check current FD rates across top banks (SBI, HDFC, ICICI, PNB) and high-yield Small Finance Banks (AU, Unity, Equitas).",
            "   - Call `get_top_indian_mutual_funds_by_category` or `search_indian_mutual_funds` to identify top direct-growth mutual funds (e.g. Flexi Cap, Large Cap Index, Balanced Advantage/Hybrid, Liquid/Debt).",
            "   - Call `get_mutual_fund_nav` to fetch real-time NAVs and details for the selected funds.",
            "   - Call `calculate_fd_maturity` and `calculate_mf_returns` to compute accurate projected returns.",
            "",
            "3. RECOMMEND A STRUCTURED ALLOCATION:",
            "   - If user asks for both FD and MF (or asks generally): Propose a balanced split (e.g., 50% FD for guaranteed safety & liquidity + 50% in Diversified Mutual Funds for inflation-beating wealth creation).",
            "   - If user asks specifically for only FD or only MF, focus on that asset class while mentioning the alternative.",
            "",
            "4. OUTPUT FORMAT:",
            "   - Structure your response clearly using markdown tables and headers:",
            "     * **Executive Summary & Recommended Allocation Plan** (Table with Asset Class, Amount in ₹, Percentage, Target Tenure, Expected Return %)",
            "     * **Part A: Bank Fixed Deposit (FD) Strategy** (Specific banks with tenure, interest rate, senior citizen rates, DICGC insurance ₹5L rule, and calculated maturity value)",
            "     * **Part B: Indian Mutual Fund Strategy** (Specific fund recommendations with category, scheme code, live NAV, rationale, and 3-5 yr projected growth)",
            "     * **Part A and part B both should be output If user asks for both FD and MF. If user asks for only FD or only MF then output only that part.",
            "     * **Taxation & Liquidity Considerations** (FD interest slab tax vs Equity MF LTCG @ 12.5% above ₹1.25L; premature withdrawal rules)",
            "     * **Actionable Next Steps** (Step-by-step how to execute via net banking or direct MF portals)",
            "",
            "5. TONE & ACCURACY:",
            "   - Professional, encouraging, unbiased, and SEBI-aware.",
            "   - Always use Indian numbering conventions (Lakhs, Crores, ₹).",
            "   - Always emphasize Direct Plans (lower expense ratios) over Regular plans.",
            "   - Include standard disclaimer: 'Mutual fund investments are subject to market risks. Past performance does not guarantee future returns.'",
        ],
        markdown=True,
    )

    return advisor_agent
