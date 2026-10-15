import os
import sys
from pathlib import Path

# Ensure src is on Python search path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
import pandas as pd

# Sync secrets to env if running in hosted cloud
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str) and _key not in os.environ:
            os.environ[_key] = _value
except Exception:
    pass

from my_invest.advisor_service import AdvisorService, AdvisoryResult
from my_invest.cache import get_semantic_cache
from my_invest.gateway import get_openrouter_account_info
from my_invest.settings import settings
from my_invest.tools.bank_fd_tool import BENCHMARK_FD_RATES, get_indian_bank_fd_rates
from my_invest.tools.cache import get_aggregated_cache_summary
from my_invest.tools.calculator_tool import calculate_fd_maturity, calculate_mf_returns
from my_invest.tools.mutual_fund_tool import (
    TOP_CURATED_MUTUAL_FUNDS,
    get_mutual_fund_nav,
    search_indian_mutual_funds,
)

st.set_page_config(
    page_title=settings.app_name,
    page_icon=":material/account_balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Financial UI
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #334155;
        color: white;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_service() -> AdvisorService:
    return AdvisorService()


if "advisory_result" not in st.session_state:
    st.session_state.advisory_result = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "session_llm_count" not in st.session_state:
    st.session_state.session_llm_count = 0
if "session_tool_count" not in st.session_state:
    st.session_state.session_tool_count = 0
if "session_total_tokens" not in st.session_state:
    st.session_state.session_total_tokens = 0
if "session_total_cost_usd" not in st.session_state:
    st.session_state.session_total_cost_usd = 0.0

# Header
st.title("My-Invest :material/account_balance:")
st.caption("AI-Powered Investment & Portfolio Advisor for Indian Bank Fixed Deposits (FDs) and Mutual Funds (MFs)")

# Sidebar Configuration
with st.sidebar:
    st.header("Advisor Settings")
    st.caption(f"🤖 **Advisor Model**: `{settings.model_name}`")
    st.caption(f"🛡️ **Guardrail Model**: `{settings.guard_model_name}`")
    st.divider()

    st.markdown("**Quick Preset Prompts**")
    presets = {
        "Balanced (₹2 Lakhs in FD & MF)": "I want to invest 2 lacs inr in Indian bank fixed deposits and in Indian mutual funds for 3 years with balanced risk.",
        "Conservative / Capital Safety (₹3 Lakhs)": "I want to invest 3 lacs inr with high safety in Indian bank FDs and short-term debt mutual funds.",
        "Wealth Growth (₹2 Lakhs - Young Professional)": "I want to invest 2 lacs inr for 5 years: 30% in high interest bank FDs and 70% in high return equity mutual funds.",
        "Senior Citizen FD Focus (₹5 Lakhs)": "I am a senior citizen looking to invest 5 lacs inr for regular interest income in safest high-rate bank FDs.",
    }
    selected_preset = st.radio("Choose a template:", list(presets.keys()), index=0)

    st.divider()
    st.info("💡 **Tip**: All recommendations prioritize **Direct Plans** (0% distributor commission) and highlight **DICGC insurance (up to ₹5 Lakhs)** on bank deposits.")

    st.divider()
    # LLM Gateway Sidebar Panel
    st.markdown("**🌐 LLM Gateway (Tier 1)**")
    gateway_icon = "🟢" if settings.gateway_enabled else "🔵"
    st.caption(f"{gateway_icon} **Provider**: `{settings.active_provider}`")
    st.caption(f"🧠 **Model**: `{settings.active_model_name}`")

    if settings.gateway_enabled:
        acct_info = get_openrouter_account_info()
        if acct_info.success:
            tier_badge = "🆓 Free Tier" if acct_info.is_free_tier else "💳 Paid Tier"
            st.caption(f"🏷️ **Account**: `{tier_badge}`")
            st.caption(f"💰 **Total Spend**: `${acct_info.usage_usd:.4f}`")
            limit_str = f"${acct_info.limit_usd:.2f}" if acct_info.limit_usd is not None else "Unlimited"
            st.caption(f"🛑 **Budget Limit**: `{limit_str}`")
            if acct_info.free_requests_limit > 0:
                st.caption(f"📊 **Free Quota**: `{acct_info.free_requests_used}/{acct_info.free_requests_limit} today`")

    st.markdown("**⚙️ Token & Model Controls**")
    selected_max_tokens = st.slider(
        "🛑 Max Output Tokens",
        min_value=400,
        max_value=3000,
        value=int(settings.max_tokens),
        step=100,
        help="Caps maximum response tokens per query to control costs and prevent runaway generations.",
    )
    selected_temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(settings.temperature),
        step=0.05,
        help="Lower values (0.1 - 0.3) provide more precise, factual financial advice. Higher values increase variability.",
    )

    st.divider()
    # Qdrant Semantic Cache Sidebar Panel
    sem_cache = get_semantic_cache()
    sem_stats = sem_cache.get_stats()
    st.markdown("**⚡ Qdrant Semantic Cache (Tier 2)**")
    mode_icon = "🟢" if "Cloud" in sem_stats["mode"] else "🔵"
    st.caption(f"{mode_icon} **Status**: `{sem_stats['mode']}`")
    st.caption(f"📦 **Collection**: `{sem_stats['collection_name']}`")
    st.caption(f"🧠 **Indexed Blueprints**: **{sem_stats['total_cached_blueprints']}**")
    st.caption(f"🎯 **Similarity Threshold**: `{sem_stats['threshold']}`")
    if sem_stats["cloud_url"] and "http" in sem_stats["cloud_url"]:
        st.caption(f"🌐 **Endpoint**: `{sem_stats['cloud_url'].split('@')[-1][:35]}...`")

# Tabs for multi-feature exploration
tab_advisor, tab_fd_rates, tab_mf_lookup, tab_calculator = st.tabs([
    "🤖 AI Portfolio Advisor",
    "🏦 Live FD Rate Explorer",
    "📈 Indian Mutual Fund NAVs",
    "🧮 Return & Maturity Calculator"
])

# ----------------- TAB 1: AI PORTFOLIO ADVISOR -----------------
with tab_advisor:
    with st.form("investment_form", border=True):
        st.subheader("What is your investment objective?")
        user_prompt = st.text_area(
            "Enter your investment details or ask freely:",
            value=presets[selected_preset],
            height=90,
            help="Specify amount, preference for FD / Mutual Funds, tenure, or risk appetite.",
        )
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_btn = st.form_submit_button("Generate Plan", type="primary", width="stretch")

    if submit_btn and user_prompt:
        try:
            service = get_service()
            with st.status("Analyzing bank FD rates, AMFI mutual funds, and computing optimal allocations...", expanded=True) as status:
                st.write("🔍 Tracking latest public, private, and small finance bank FD rates...")
                st.write("📊 Querying AMFI database for direct growth mutual funds & NAVs...")
                st.write("🧮 Calculating compound maturity and tax projections...")
                result = service.run_advisory(
                    user_prompt,
                    temperature=selected_temperature,
                    max_tokens=selected_max_tokens,
                )
                st.session_state.advisory_result = result
                st.session_state.session_llm_count += result.llm_call_count
                st.session_state.session_tool_count += result.tool_call_count
                st.session_state.session_total_tokens += result.total_tokens
                st.session_state.session_total_cost_usd += getattr(result, "estimated_cost_usd", 0.0)
                st.session_state.query_history.append({
                    "prompt": user_prompt,
                    "duration_s": result.duration_ms / 1000,
                    "llm_calls": result.llm_call_count,
                    "tool_count": result.tool_call_count,
                    "tools_used": list(result.tools_used),
                    "tokens": result.total_tokens,
                    "cost_usd": getattr(result, "estimated_cost_usd", 0.0),
                    "model": result.model_name,
                })
                status.update(label="Investment plan ready!", state="complete", expanded=False)
        except Exception as e:
            st.error(f"Error generating investment strategy: {e}")

    res: AdvisoryResult | None = st.session_state.advisory_result
    if res:
        with st.container(border=True):
            col_meta1, col_meta2 = st.columns([3, 2])
            with col_meta1:
                st.subheader("Your Personalized Investment Blueprint")
            with col_meta2:
                tools_badge = f"{res.tool_call_count} tool{'s' if res.tool_call_count != 1 else ''}"
                llm_badge = f"{res.llm_call_count} LLM call{'s' if res.llm_call_count != 1 else ''}"
                st.caption(f"⚡ {res.duration_ms / 1000:.1f}s | 🤖 {llm_badge} | 🛠️ {tools_badge}")

            if getattr(res, "is_cached", False):
                st.success(
                    f"⚡ **Served by Qdrant Semantic Cache** (Similarity Score: **{res.similarity_score * 100:.1f}%**) — **0 LLM inferences** and **0 tool calls** consumed! Delivered in {res.duration_ms:.0f}ms."
                )
            elif getattr(res, "is_guarded", False):
                guard_label = getattr(res, "model_name", "Input Guardrail")
                st.warning(
                    f"🛡️ **Guardrail Active ({guard_label})**: Query filtered or redirected. Handled in {res.duration_ms:.0f}ms with **0 LLM calls** and **0 tool calls**."
                )

            st.markdown(res.text)

            if not getattr(res, "is_guarded", False):
                st.divider()
                st.download_button(
                    "Download Strategy Report (.md)",
                    data=res.text,
                    file_name="my_invest_strategy_report.md",
                    mime="text/markdown",
                    icon=":material/download:",
                )
    else:
        with st.container(border=True):
            st.info("👈 Enter your investment goal above or click **Generate Plan** to receive a tailored portfolio allocation.")

# ----------------- TAB 2: LIVE FD RATE EXPLORER -----------------
with tab_fd_rates:
    st.subheader("Current Fixed Deposit (FD) Rates Across Indian Banks")
    st.caption("Interest compounded quarterly. Bank deposits insured up to ₹5,00,000 per depositor per bank by DICGC (RBI subsidiary).")

    fd_category = st.selectbox("Select Bank Category:", list(BENCHMARK_FD_RATES.keys()), index=0)
    bank_data = BENCHMARK_FD_RATES[fd_category]
    df_banks = pd.DataFrame(bank_data)
    st.dataframe(df_banks, width="stretch", hide_index=True)

    with st.expander("Search Live Updates from the Web"):
        search_query = st.text_input("Search bank or tenure (e.g., 'SBI 400 days Amrit Kalash' or 'HDFC 35 months'):", "SBI HDFC FD rates 2025")
        if st.button("Search Online Rates"):
            with st.spinner("Searching online for live rate updates..."):
                rates_text = get_indian_bank_fd_rates(search_query)
                st.markdown(rates_text)

# ----------------- TAB 3: INDIAN MUTUAL FUND NAVS -----------------
with tab_mf_lookup:
    st.subheader("Official AMFI Indian Mutual Fund Directory & Live NAV Lookup")
    st.caption("Directly connected to the AMFI NAV engine via `api.mfapi.in`.")

    mf_col1, mf_col2 = st.columns([2, 1])
    with mf_col1:
        mf_search_input = st.text_input("Search any fund by name or AMC (e.g. Parag Parikh, UTI Nifty, HDFC Balanced, Quant):", "Parag Parikh Flexi Cap")
    with mf_col2:
        st.write("")
        st.write("")
        do_search = st.button("Search AMFI Funds", type="secondary", width="stretch")

    if do_search or mf_search_input:
        with st.spinner(f"Searching AMFI database for '{mf_search_input}'..."):
            search_out = search_indian_mutual_funds(mf_search_input)
            st.markdown(search_out)

    st.divider()
    st.subheader("Curated Top Funds by Category")
    for cat_name, fund_list in TOP_CURATED_MUTUAL_FUNDS.items():
        with st.expander(cat_name):
            for f in fund_list:
                col_f1, col_f2 = st.columns([3, 1])
                with col_f1:
                    st.markdown(f"**{f['name']}**  \n*Scheme Code: `{f['scheme_code']}` | Risk: {f['risk']}*")
                with col_f2:
                    if st.button(f"Fetch Live NAV", key=f"nav_{f['scheme_code']}"):
                        nav_info = get_mutual_fund_nav(f["scheme_code"])
                        st.info(nav_info)

# ----------------- TAB 4: CALCULATOR -----------------
with tab_calculator:
    st.subheader("Financial Return & Maturity Calculator")
    calc_type = st.radio("Choose calculation type:", ["Bank Fixed Deposit (FD)", "Mutual Fund Lumpsum (CAGR)"], horizontal=True)

    if calc_type == "Bank Fixed Deposit (FD)":
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            fd_principal = st.number_input("Deposit Amount (INR ₹):", min_value=5000.0, value=100000.0, step=10000.0)
        with col_c2:
            fd_rate = st.number_input("Interest Rate (% p.a.):", min_value=3.0, max_value=12.0, value=7.25, step=0.1)
        with col_c3:
            fd_years = st.number_input("Tenure (Years):", min_value=0.25, max_value=10.0, value=3.0, step=0.25)

        calc_result = calculate_fd_maturity(fd_principal, fd_rate, fd_years)
        st.success(calc_result)

    else:
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            mf_amount = st.number_input("Lumpsum Investment (INR ₹):", min_value=5000.0, value=100000.0, step=10000.0)
        with col_m2:
            mf_cagr = st.number_input("Expected Annual CAGR (%):", min_value=4.0, max_value=25.0, value=12.0, step=0.5)
        with col_m3:
            mf_years = st.number_input("Investment Horizon (Years):", min_value=1.0, max_value=25.0, value=5.0, step=1.0)

        calc_result = calculate_mf_returns(mf_amount, mf_cagr, mf_years)
        st.success(calc_result)

# ----------------- TELEMETRY & USAGE DASHBOARD (BOTTOM OF PAGE) -----------------
st.divider()
st.subheader("⚡ Live AI & Tool Execution Telemetry")
st.caption(f"Real-time counter and audit log of {settings.active_provider} inferences and financial tools executed per query.")

current_res: AdvisoryResult | None = st.session_state.advisory_result
if current_res:
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    with t_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🤖 LLM Inferences (Latest Query)</div>
                <div class="metric-value">{current_res.llm_call_count} <span style="font-size: 1rem; color: #94a3b8;">call{'s' if current_res.llm_call_count != 1 else ''}</span></div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Model: <code>{current_res.model_name.split('/')[-1]}</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🛠️ Tools Used (Latest Query)</div>
                <div class="metric-value">{current_res.tool_call_count} <span style="font-size: 1rem; color: #94a3b8;">call{'s' if current_res.tool_call_count != 1 else ''}</span></div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Unique: {len(current_res.tools_used)} distinct tool{'s' if len(current_res.tools_used) != 1 else ''}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t_col3:
        cost_str = f"~${current_res.estimated_cost_usd:.5f}" if current_res.estimated_cost_usd > 0 else "Free / $0.00"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🪙 Tokens & Cost (Latest)</div>
                <div class="metric-value">{current_res.total_tokens:,} <span style="font-size: 1rem; color: #94a3b8;">tokens</span></div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">In: {current_res.input_tokens:,} | Out: {current_res.output_tokens:,} | Est: <code>{cost_str}</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t_col4:
        query_total = len(st.session_state.query_history)
        session_cost_str = f"~${st.session_state.session_total_cost_usd:.4f}"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">📈 Session Cumulative</div>
                <div class="metric-value">{query_total} <span style="font-size: 1rem; color: #94a3b8;">quer{'ies' if query_total != 1 else 'y'}</span></div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">{st.session_state.session_total_tokens:,} tokens ({session_cost_str}) | {st.session_state.session_tool_count} tools</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("**Tools Executed in Latest Query:**")
    if current_res.tools_used:
        tool_icons = {
            "get_indian_bank_fd_rates": ("🏦", "Bank FD Rate Tracker"),
            "search_indian_mutual_funds": ("🔍", "AMFI Fund Search"),
            "get_mutual_fund_nav": ("📈", "AMFI Live NAV Fetcher"),
            "get_top_indian_mutual_funds_by_category": ("⭐", "Curated MF Directory"),
            "get_indian_etf_price": ("📊", "Yahoo Finance ETF Tracker"),
            "calculate_fd_maturity": ("🧮", "FD Quarterly Compounding"),
            "calculate_mf_returns": ("📈", "MF CAGR Wealth Calculator"),
            "calculate_portfolio_split": ("💼", "Corpus Asset Allocator"),
            "duckduckgo_search": ("🌐", "DuckDuckGo Web Search"),
            "yfinance": ("💹", "YFinance Live Market"),
        }
        badges = []
        for tool_name in current_res.tools_used:
            icon, label = tool_icons.get(tool_name, ("⚙️", tool_name))
            badges.append(
                f'<span style="background-color: #1e293b; color: #38bdf8; border: 1px solid #334155; padding: 4px 10px; border-radius: 16px; margin-right: 8px; margin-bottom: 6px; display: inline-block; font-size: 0.85rem; font-weight: 500;">{icon} <b>{tool_name}</b> ({label})</span>'
            )
        st.markdown("".join(badges), unsafe_allow_html=True)
    else:
        st.info("ℹ️ **Direct Synthesis**: The LLM resolved the query directly without requiring external tool calls.")

    # Tier 1 Tool Cache Live Telemetry
    cache_summary = get_aggregated_cache_summary()
    sem_stats = get_semantic_cache().get_stats()
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 8px; padding: 10px 14px; margin-top: 14px; margin-bottom: 12px;">
                <div style="color: #38bdf8; font-weight: 700; font-size: 0.9rem;">⚡ Tier 1: Tool TTL Cache</div>
                <div style="color: #f8fafc; font-weight: 600; font-size: 1.1rem; margin-top: 2px;">{cache_summary['total_hits']} Hits ({cache_summary['hit_rate_pct']}% hit rate)</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">{cache_summary['total_misses']} live network fetches | &lt;1ms cached execution</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_c2:
        sem_mode_badge = "Cloud" if "Cloud" in sem_stats["mode"] else "Disk"
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 8px; padding: 10px 14px; margin-top: 14px; margin-bottom: 12px;">
                <div style="color: #a855f7; font-weight: 700; font-size: 0.9rem;">🧠 Tier 2: Qdrant Semantic Cache ({sem_mode_badge})</div>
                <div style="color: #f8fafc; font-weight: 600; font-size: 1.1rem; margin-top: 2px;">{sem_stats['hits']} Hits ({sem_stats['hit_rate_pct']}% savings)</div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">{sem_stats['total_cached_blueprints']} vector blueprints indexed | Cosine threshold: {sem_stats['threshold']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander(f"📜 View Session Query History & Tool Audit Trail ({len(st.session_state.query_history)} queries executed)"):
        # Tool Cache breakdown
        if cache_summary.get("caches"):
            st.markdown("**Tool Cache Breakdown (Tier 1 TTL Engine):**")
            cache_rows = []
            for name, c in cache_summary["caches"].items():
                ttl_str = f"{c.ttl_seconds // 3600}h" if c.ttl_seconds >= 3600 else f"{c.ttl_seconds // 60}m"
                cache_rows.append({
                    "Tool Name": name,
                    "Hits (Instant)": c.hits,
                    "Misses (Network)": c.misses,
                    "Hit Rate": f"{c.hit_rate_pct}%",
                    "Entries in Cache": f"{c.currsize}/{c.maxsize}",
                    "TTL Expiry": ttl_str,
                })
            st.dataframe(pd.DataFrame(cache_rows), width="stretch", hide_index=True)
            st.divider()

        for i, q in enumerate(reversed(st.session_state.query_history), 1):
            q_num = len(st.session_state.query_history) - i + 1
            st.markdown(f"**Query #{q_num}**: *\"{q['prompt'][:140]}{'...' if len(q['prompt']) > 140 else ''}\"*")
            st.caption(f"🤖 LLM Calls: **{q['llm_calls']}** | 🛠️ Tools Triggered: **{q['tool_count']}** | ⏱️ Duration: **{q['duration_s']:.2f}s** | Model: `{q['model']}`")
            if q["tools_used"]:
                st.code("Tools: " + ", ".join(q["tools_used"]), language="text")
            st.divider()

else:
    st.markdown(
        """
        <div style="background: #0f172a; border: 1px dashed #334155; border-radius: 10px; padding: 1.2rem; text-align: center; color: #94a3b8;">
            <div style="font-size: 1.1rem; color: #cbd5e1; font-weight: 600; margin-bottom: 4px;">No Queries Executed Yet</div>
            <div style="font-size: 0.9rem;">Submit an investment goal in the <b>AI Portfolio Advisor</b> tab above to view real-time telemetry: LLM inference counters, tools invoked, and execution latencies.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.caption("⚠️ **Regulatory Disclaimer**: AI-generated investment plans and projections are strictly for educational and research purposes. Please consult a SEBI-registered Investment Advisor (RIA) before making financial decisions.")
