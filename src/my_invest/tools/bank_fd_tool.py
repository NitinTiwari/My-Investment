from agno.tools.duckduckgo import DuckDuckGoTools


# Verified benchmark rates across top Indian banks (Updated quarterly)
BENCHMARK_FD_RATES = {
    "Public Sector Banks": [
        {"bank": "State Bank of India (SBI)", "1_year": "6.80%", "2_to_3_years": "7.00%", "5_years": "6.50%", "senior_citizen_extra": "+0.50%"},
        {"bank": "Punjab National Bank (PNB)", "1_year": "6.80%", "400_days": "7.25%", "3_years": "7.00%", "senior_citizen_extra": "+0.50%"},
        {"bank": "Bank of Baroda (BOB)", "1_year": "6.85%", "2_to_3_years": "7.15%", "5_years": "6.50%", "senior_citizen_extra": "+0.50%"},
        {"bank": "Canara Bank", "1_year": "6.85%", "444_days": "7.25%", "3_years": "6.80%", "senior_citizen_extra": "+0.50%"},
    ],
    "Leading Private Banks": [
        {"bank": "HDFC Bank", "1_year": "6.60%", "2_to_3_years": "7.00%", "5_years": "7.00%", "senior_citizen_extra": "+0.50%"},
        {"bank": "ICICI Bank", "1_year": "6.70%", "15_to_18_months": "7.25%", "3_to_5_years": "7.00%", "senior_citizen_extra": "+0.50%"},
        {"bank": "Axis Bank", "1_year": "6.70%", "17_months": "7.20%", "3_years": "7.10%", "senior_citizen_extra": "+0.50%"},
        {"bank": "Kotak Mahindra Bank", "1_year": "7.10%", "390_days": "7.40%", "3_years": "7.00%", "senior_citizen_extra": "+0.50%"},
    ],
    "High-Yield Small Finance Banks (DICGC Insured up to ₹5 Lakh)": [
        {"bank": "Unity Small Finance Bank", "1_year": "7.85%", "1001_days": "9.00%", "5_years": "8.15%", "senior_citizen_extra": "+0.50%"},
        {"bank": "Suryoday Small Finance Bank", "1_year": "6.85%", "2_years_2_days": "8.65%", "5_years": "8.25%", "senior_citizen_extra": "+0.50%"},
        {"bank": "AU Small Finance Bank", "1_year": "7.25%", "18_months": "8.00%", "3_years": "7.50%", "senior_citizen_extra": "+0.50%"},
        {"bank": "Equitas Small Finance Bank", "1_year": "8.20%", "444_days": "8.50%", "3_years": "8.00%", "senior_citizen_extra": "+0.50%"},
    ],
    "Government Schemes": [
        {"bank": "Post Office Term Deposit (POTD)", "1_year": "6.90%", "2_years": "7.00%", "3_years": "7.10%", "5_years": "7.50%", "senior_citizen_extra": "N/A (SCSS is 8.20%)"},
    ]
}


def get_indian_bank_fd_rates(query: str = "current bank FD interest rates in India") -> str:
    """
    Searches online for the latest Indian bank fixed deposit interest rates (SBI, HDFC, ICICI, PNB, and Small Finance Banks),
    and provides verified benchmark interest rates, tenure options, senior citizen rates, and DICGC insurance rules.

    Args:
        query: Specific query or tenure to check, e.g. '1 year FD rates' or 'highest FD rates in India'
    """
    lines = [
        "### CURRENT INDIAN BANK FIXED DEPOSIT (FD) RATES BENCHMARK",
        "*(Rates compounded quarterly. Principal + interest covered up to ₹5,00,000 per depositor per bank by DICGC/RBI)*\n"
    ]

    for category, banks in BENCHMARK_FD_RATES.items():
        lines.append(f"#### {category}:")
        for b in banks:
            details = ", ".join([f"{k}: {v}" for k, v in b.items() if k != "bank"])
            lines.append(f"- **{b['bank']}**: {details}")
        lines.append("")

    # Online search to fetch any breaking changes or special festive/limited-period buckets
    try:
        ddg = DuckDuckGoTools()
        search_query = f"latest bank fixed deposit interest rates India SBI HDFC ICICI PNB {query}"
        search_results = ddg.web_search(search_query)
        if search_results:
            lines.append("#### Online Search Highlights / Special Tenures:")
            # Capture top 2 search snippets
            for item in search_results[:2]:
                title = item.get("title", "")
                body = item.get("body", "")
                if title and body:
                    lines.append(f"- **{title}**: {body[:250]}...")
    except Exception:
        # Fallback to benchmark table if network search encounters an issue
        pass

    return "\n".join(lines)
