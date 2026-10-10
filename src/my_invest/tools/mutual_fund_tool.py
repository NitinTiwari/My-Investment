import requests
import yfinance as yf
from my_invest.tools.cache import ttl_cache

# Curated reference directory of top-rated, consistent Indian Mutual Funds across AMFI categories
TOP_CURATED_MUTUAL_FUNDS = {
    "Flexi Cap (Wealth Creation, 5+ yrs)": [
        {"name": "Parag Parikh Flexi Cap Fund - Direct Plan - Growth", "scheme_code": 122639, "category": "Flexi Cap", "risk": "Very High"},
        {"name": "HDFC Flexi Cap Fund - Direct Plan - Growth", "scheme_code": 118989, "category": "Flexi Cap", "risk": "Very High"},
        {"name": "JM Flexicap Fund - (Direct) - Growth Option", "scheme_code": 120505, "category": "Flexi Cap", "risk": "Very High"},
    ],
    "Large Cap & Index (Core Stability, 3-5 yrs)": [
        {"name": "UTI Nifty 50 Index Fund - Direct Plan - Growth Option", "scheme_code": 120716, "category": "Index Fund", "risk": "Very High"},
        {"name": "ICICI Prudential Bluechip Fund - Direct Plan - Growth", "scheme_code": 120586, "category": "Large Cap", "risk": "Very High"},
        {"name": "HDFC Top 100 Fund - Direct Plan - Growth", "scheme_code": 118987, "category": "Large Cap", "risk": "Very High"},
    ],
    "Hybrid & Balanced Advantage (Moderate Risk, 3+ yrs)": [
        {"name": "HDFC Balanced Advantage Fund - Direct Plan - Growth", "scheme_code": 118968, "category": "Dynamic Asset Allocation / Balanced", "risk": "High"},
        {"name": "ICICI Prudential Balanced Advantage Fund - Direct Plan - Growth", "scheme_code": 120302, "category": "Balanced Advantage", "risk": "High"},
        {"name": "Edelweiss Balanced Advantage Fund - Direct Plan - Growth", "scheme_code": 135781, "category": "Balanced Advantage", "risk": "High"},
    ],
    "Debt & Liquid (Capital Safety & High Liquidity, 1-3 yrs)": [
        {"name": "SBI Liquid Fund - Direct Plan - Growth", "scheme_code": 119842, "category": "Liquid Fund", "risk": "Low to Moderate"},
        {"name": "ICICI Prudential Corporate Bond Fund - Direct Plan - Growth", "scheme_code": 120387, "category": "Corporate Bond Fund", "risk": "Moderate"},
        {"name": "HDFC Short Term Debt Fund - Direct Plan - Growth", "scheme_code": 118973, "category": "Short Duration Debt", "risk": "Moderate"},
    ],
    "Mid & Small Cap (Aggressive Growth, 7+ yrs)": [
        {"name": "Motilal Oswal Midcap Fund - Direct Plan - Growth", "scheme_code": 127042, "category": "Mid Cap", "risk": "Very High"},
        {"name": "Nippon India Small Cap Fund - Direct Plan - Growth", "scheme_code": 118778, "category": "Small Cap", "risk": "Very High"},
    ]
}


@ttl_cache(maxsize=256, ttl_seconds=86400)
def search_indian_mutual_funds(query: str) -> str:
    """
    Searches the official AMFI mutual fund database for Indian mutual funds matching the query.
    Returns matched scheme names with their unique AMFI schemeCode for direct NAV lookup.

    Args:
        query: Name or keywords of the fund (e.g. 'Parag Parikh', 'Nifty Index', 'HDFC Balanced', 'SBI Bluechip')
    """
    url = f"https://api.mfapi.in/mf/search?q={requests.utils.quote(query)}"
    try:
        response = requests.get(url, timeout=8)
        if response.status_code == 200:
            schemes = response.json()
            if not schemes:
                return f"No mutual funds found matching '{query}'. Try broader terms like 'HDFC', 'SBI', 'Nifty', or 'Flexi'."

            # Prioritize Direct Growth schemes for retail investor cost efficiency
            direct_growth = [s for s in schemes if "direct" in s["schemeName"].lower() and "growth" in s["schemeName"].lower()]
            selected = direct_growth[:5] if direct_growth else schemes[:5]

            results = [f"Found {len(schemes)} schemes matching '{query}'. Top direct matches:"]
            for s in selected:
                results.append(f"- Scheme Code: `{s['schemeCode']}` | Name: **{s['schemeName']}**")
            return "\n".join(results)
    except Exception as e:
        return f"Error querying AMFI mutual fund search: {e}"
    return "Could not retrieve mutual fund data."


@ttl_cache(maxsize=512, ttl_seconds=43200)
def get_mutual_fund_nav(scheme_code: int) -> str:
    """
    Fetches the latest official Net Asset Value (NAV), fund house, scheme category,
    and date for a specific Indian mutual fund from AMFI using its schemeCode.

    Args:
        scheme_code: The AMFI scheme code (e.g., 122639 for Parag Parikh Flexi Cap Direct Growth)
    """
    url = f"https://api.mfapi.in/mf/{scheme_code}/latest"
    try:
        response = requests.get(url, timeout=8)
        if response.status_code == 200:
            data = response.json()
            meta = data.get("meta", {})
            data_points = data.get("data", [])
            latest = data_points[0] if data_points else {}

            return (
                f"**Scheme Name**: {meta.get('scheme_name')}\n"
                f"- **Fund House**: {meta.get('fund_house')}\n"
                f"- **Category**: {meta.get('scheme_category')}\n"
                f"- **Scheme Code**: {meta.get('scheme_code')}\n"
                f"- **Latest NAV**: ₹{latest.get('nav', 'N/A')} (as of {latest.get('date', 'N/A')})"
            )
    except Exception as e:
        return f"Error fetching NAV for scheme {scheme_code}: {e}"
    return f"Unable to fetch NAV for scheme {scheme_code}."


@ttl_cache(maxsize=32, ttl_seconds=86400)
def get_top_indian_mutual_funds_by_category(category: str = "all") -> str:
    """
    Returns a curated list of top-performing, highly rated Indian Mutual Funds across categories
    (Flexi Cap, Large Cap/Index, Hybrid/Balanced Advantage, Debt/Liquid, Small/Mid Cap)
    with their AMFI scheme codes.

    Args:
        category: 'all', 'flexi cap', 'large cap', 'hybrid', 'debt', or 'small cap'
    """
    lines = ["### TOP CURATED INDIAN MUTUAL FUNDS BY CATEGORY (Direct Growth Plans)"]
    cat_lower = category.lower()

    for cat_title, funds in TOP_CURATED_MUTUAL_FUNDS.items():
        if cat_lower != "all" and cat_lower not in cat_title.lower():
            continue
        lines.append(f"\n#### {cat_title}:")
        for f in funds:
            lines.append(f"- **{f['name']}**")
            lines.append(f"  Scheme Code: `{f['scheme_code']}` | Risk: {f['risk']}")

    lines.append("\n*Note: Direct Plans have lower expense ratios (0.5% - 1.5% lower) compared to Regular Plans, yielding higher returns over time.*")
    return "\n".join(lines)


@ttl_cache(maxsize=64, ttl_seconds=900)
def get_indian_etf_price(ticker: str) -> str:
    """
    Fetches live market price and details of Indian Exchange Traded Funds (ETFs)
    listed on NSE/BSE (e.g., NIFTYBEES.NS, GOLDBEES.NS, JUNIORBEES.NS, BANKBEES.NS).

    Args:
        ticker: The Indian ETF symbol with exchange suffix (e.g. 'NIFTYBEES.NS', 'GOLDBEES.NS')
    """
    ticker_clean = ticker.upper().strip()
    if not (ticker_clean.endswith(".NS") or ticker_clean.endswith(".BO")):
        ticker_clean = f"{ticker_clean}.NS"

    try:
        t = yf.Ticker(ticker_clean)
        price = t.fast_info.get("last_price") or t.info.get("regularMarketPrice")
        currency = t.fast_info.get("currency", "INR")
        name = t.info.get("shortName") or ticker_clean
        return f"**{name}** ({ticker_clean}): {currency} {price:.2f}"
    except Exception as e:
        return f"Error fetching ETF price for {ticker}: {e}"
