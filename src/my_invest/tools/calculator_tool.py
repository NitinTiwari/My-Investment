from my_invest.tools.cache import ttl_cache


@ttl_cache(maxsize=128, ttl_seconds=3600)
def calculate_fd_maturity(
    principal_inr: float,
    annual_rate_pct: float,
    tenure_years: float,
    compounding_frequency: int = 4
) -> str:
    """
    Calculates the exact maturity amount and interest earned for an Indian Bank Fixed Deposit.
    Indian banks compound interest quarterly (4 times per year).

    Args:
        principal_inr: Deposit amount in INR (e.g., 100000)
        annual_rate_pct: Annual interest rate in percent (e.g., 7.25 for 7.25%)
        tenure_years: Investment tenure in years (e.g., 1.0, 2.5, 3.0, 5.0)
        compounding_frequency: Number of compounding periods per year (default 4 for quarterly)
    """
    try:
        P = float(principal_inr)
        r = float(annual_rate_pct) / 100.0
        n = int(compounding_frequency)
        t = float(tenure_years)

        # Compound interest formula: A = P * (1 + r/n)^(n*t)
        A = P * ((1.0 + (r / n)) ** (n * t))
        interest_earned = A - P
        effective_annual_yield = ((A / P) ** (1.0 / t) - 1.0) * 100.0 if t > 0 else annual_rate_pct

        return (
            f"**Fixed Deposit (FD) Maturity Calculation**:\n"
            f"- **Principal Deposited**: ₹{P:,.2f}\n"
            f"- **Interest Rate**: {annual_rate_pct:.2f}% p.a. (Compounded Quarterly)\n"
            f"- **Tenure**: {tenure_years} year(s)\n"
            f"- **Total Maturity Value**: ₹{A:,.2f}\n"
            f"- **Total Interest Earned**: ₹{interest_earned:,.2f}\n"
            f"- **Effective Annual Yield**: {effective_annual_yield:.2f}%"
        )
    except Exception as e:
        return f"Error calculating FD maturity: {e}"


@ttl_cache(maxsize=128, ttl_seconds=3600)
def calculate_mf_returns(
    lumpsum_inr: float,
    expected_cagr_pct: float,
    tenure_years: float
) -> str:
    """
    Calculates the projected future wealth and capital gains for a lumpsum Mutual Fund investment
    based on expected compound annual growth rate (CAGR).

    Args:
        lumpsum_inr: Lumpsum investment amount in INR (e.g., 100000)
        expected_cagr_pct: Expected annual CAGR in percent (e.g., 12.0 for equity, 7.5 for debt/hybrid)
        tenure_years: Investment horizon in years (e.g., 3, 5, 7)
    """
    try:
        P = float(lumpsum_inr)
        r = float(expected_cagr_pct) / 100.0
        t = float(tenure_years)

        # Compound wealth formula: A = P * (1 + r)^t
        A = P * ((1.0 + r) ** t)
        wealth_gain = A - P

        return (
            f"**Mutual Fund Projected Growth (CAGR Projection)**:\n"
            f"- **Initial Investment**: ₹{P:,.2f}\n"
            f"- **Assumed CAGR**: {expected_cagr_pct:.1f}% p.a.\n"
            f"- **Investment Horizon**: {tenure_years} year(s)\n"
            f"- **Projected Future Value**: ₹{A:,.2f}\n"
            f"- **Estimated Capital Gain**: ₹{wealth_gain:,.2f}\n"
            f"*(Note: Mutual fund returns are subject to market risks and not guaranteed)*"
        )
    except Exception as e:
        return f"Error calculating Mutual Fund returns: {e}"


@ttl_cache(maxsize=128, ttl_seconds=3600)
def calculate_portfolio_split(
    total_corpus_inr: float,
    fd_ratio_pct: float = 50.0,
    mf_ratio_pct: float = 50.0
) -> str:
    """
    Computes exact allocation amounts in INR for a given total corpus and percentage split.

    Args:
        total_corpus_inr: Total amount available to invest in INR (e.g., 200000)
        fd_ratio_pct: Percentage to allocate to Bank FDs (e.g., 50)
        mf_ratio_pct: Percentage to allocate to Mutual Funds (e.g., 50)
    """
    total = float(total_corpus_inr)
    fd_amt = total * (fd_ratio_pct / 100.0)
    mf_amt = total * (mf_ratio_pct / 100.0)
    return (
        f"**Corpus Allocation Split**:\n"
        f"- Total Investment Corpus: ₹{total:,.2f}\n"
        f"- Bank Fixed Deposits ({fd_ratio_pct:.0f}%): ₹{fd_amt:,.2f}\n"
        f"- Mutual Funds ({mf_ratio_pct:.0f}%): ₹{mf_amt:,.2f}"
    )
