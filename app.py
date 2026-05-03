import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import ast
import re
import traceback
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(
    page_title="RetailIQ — Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "page" not in st.session_state:
    st.session_state.page = "welcome"

st.markdown("""
<style>
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; font-size: 0.95rem; }
[data-testid="stSidebar"] hr { border-color: #334155; }
.kpi-card { background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155; border-radius: 12px; padding: 18px 22px; text-align: center; height: 110px; display: flex; flex-direction: column; justify-content: center; }
.kpi-label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #f1f5f9; line-height: 1; }
.kpi-sub   { font-size: 0.72rem; color: #64748b; margin-top: 4px; }
.kpi-gold  .kpi-value { color: #fbbf24; }
.kpi-blue  .kpi-value { color: #60a5fa; }
.kpi-slate .kpi-value { color: #94a3b8; }
.kpi-green .kpi-value { color: #34d399; }
.page-title { font-size: 1.85rem; font-weight: 700; color: #f1f5f9; margin: 0; }
.page-subtitle { font-size: 0.9rem; color: #64748b; margin: 2px 0 0 0; }
.section-header { font-size: 1.05rem; font-weight: 600; color: #cbd5e1; border-left: 3px solid #3b82f6; padding-left: 10px; margin: 18px 0 10px 0; }
.pred-high { background:#052e16; border:1px solid #166534; border-radius:10px; padding:12px 18px; color:#4ade80; font-weight:600; }
.pred-low  { background:#0f172a; border:1px solid #334155; border-radius:10px; padding:12px 18px; color:#94a3b8; font-weight:600; }
.info-box  { background:#0f172a; border:1px solid #334155; border-radius:10px; padding:14px 18px; color:#94a3b8; font-size:0.88rem; }
.divider   { border:none; border-top:1px solid #1e293b; margin:20px 0; }
.tip-chip  { display:inline-block; background:#1e293b; border:1px solid #334155; color:#94a3b8; border-radius:20px; padding:4px 12px; font-size:0.8rem; margin:3px; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#0f172a", "axes.facecolor": "#0f172a",
    "axes.edgecolor": "#334155", "axes.labelcolor": "#94a3b8",
    "xtick.color": "#64748b", "ytick.color": "#64748b",
    "text.color": "#cbd5e1", "grid.color": "#1e293b",
    "grid.linestyle": "--", "grid.linewidth": 0.6,
})

BLUE = "#3b82f6"; GOLD = "#f59e0b"; SLATE = "#64748b"
GREEN = "#10b981"; PURPLE = "#8b5cf6"; CORAL = "#f43f5e"
SEG_COLORS = {"High-Value Customers": GOLD, "Regular Customers": BLUE, "Low-Value Customers": SLATE}

@st.cache_data
def load_data():
    segments = pd.read_csv("customer_segments.csv")
    cleaned  = pd.read_csv("cleaned_data.csv", parse_dates=["InvoiceDate"])
    rfm      = pd.read_csv("rfm_features.csv")
    recs_raw = pd.read_csv("recommendations.csv")
    model    = joblib.load("model.pkl")
    return segments, cleaned, rfm, recs_raw, model

segments, cleaned, rfm, recs_raw, model = load_data()
if "Revenue" not in cleaned.columns:
    cleaned["Revenue"] = cleaned["Quantity"] * cleaned["UnitPrice"]

@st.cache_data
def parse_rules(recs_raw):
    rules = []
    for _, row in recs_raw.iterrows():
        try:
            ant = ast.literal_eval(row["antecedents"])
            con = ast.literal_eval(row["consequents"])
            rules.append({"antecedents": frozenset(ant), "consequents": frozenset(con),
                          "confidence": float(row["confidence"]), "lift": float(row["lift"])})
        except Exception:
            pass
    return rules

rules = parse_rules(recs_raw)

def get_recommendations(purchased_products, rules, top_n=5):
    purchased = set(purchased_products)
    scored = {}
    for rule in rules:
        if rule["antecedents"].issubset(purchased):
            for item in rule["consequents"]:
                if item not in purchased:
                    if item not in scored or rule["confidence"] > scored[item]:
                        scored[item] = rule["confidence"]
    return sorted(scored.items(), key=lambda x: x[1], reverse=True)[:top_n]

def kpi_card(label, value, sub="", style=""):
    return f'<div class="kpi-card {style}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>'

def page_header(title, subtitle=""):
    st.markdown(f'<p class="page-title">{title}</p><p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)
    st.markdown("")

def section(label):
    st.markdown(f'<div class="section-header">{label}</div>', unsafe_allow_html=True)

def dark_fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0f172a"); ax.set_facecolor("#0f172a")
    return fig, ax

def build_customer_context(cid, row, recs, prediction):
    rec_list = ", ".join([r[0] for r in recs]) if recs else "none found"
    cust_products = cleaned[cleaned["CustomerID"] == cid]["Description"].dropna().unique().tolist()
    top_products = ", ".join(cust_products[:10]) if cust_products else "none"
    return f"""You are a retail analytics assistant with access to customer data.
Customer ID: {cid}
Segment: {row['Segment']}
Recency: {int(row['Recency'])} days since last purchase
Frequency: {int(row['Frequency'])} total orders
Monetary Value: £{float(row['Monetary']):,.2f} total spent
High-Value Prediction: {"Yes" if prediction == 1 else "No"}
Top purchased products: {top_products}
Recommended products: {rec_list}
Answer questions about this customer helpfully and concisely. Give specific, actionable business insights."""

def is_data_question(question):
    q = question.lower().strip()
    conversational_patterns = [
        r"^(hi|hello|hey|thanks|thank you|ok|okay|yes|no|sure|great|cool|nice|good|got it)[\s!?.]*$",
        r"^can i (ask|use|try)",
        r"^(what (is|are) (this|that|you|the app|rfm|segmentation|churn|recency|frequency|monetary))",
        r"^(how does (this|that|it|the app) work)",
        r"^(explain|what do you mean|what does \w+ mean)",
        r"^(are (there|you)|is (there|this|that))",
        r"^(who (are|is)|where (is|are))",
        r"^(tell me about (rfm|segmentation|clustering|machine learning|retail))",
    ]
    for pattern in conversational_patterns:
        if re.search(pattern, q): return False
    data_keywords = [
        "how many","show me","list","top","best","worst","average","total","revenue","customer",
        "product","segment","country","percent","%","chart","plot","graph","trend","compare",
        "breakdown","filter","bought","purchased","spent","orders","inactive","churn","at risk",
        "high value","low value","regular","monthly","weekly","daily","most","least","highest",
        "lowest","more than","less than","between","who","which","what is the","what are the","find","give me",
    ]
    for kw in data_keywords:
        if kw in q: return True
    return len(q.split()) > 5

def answer_general_question(question, chat_history):
    messages = [{"role": "system", "content": (
        "You are a helpful retail analytics assistant for a customer analytics app. "
        "The app analyses customer data using RFM segmentation, product recommendations, "
        "and purchase prediction. Answer general questions conversationally and helpfully. "
        "Keep answers concise — 2-3 sentences max."
    )}]
    for msg in chat_history[-4:]: messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=200)
    return response.choices[0].message.content.strip()

SCHEMA_PROMPT = """You are a Python data analyst assistant. You have access to these pandas DataFrames:

1. `segments` — Columns: CustomerID (int), Recency (int), Frequency (int), Monetary (float), Cluster (int), Segment (str: 'High-Value Customers', 'Regular Customers', 'Low-Value Customers')
2. `cleaned` — Columns: InvoiceNo, StockCode, Description, Quantity (int), InvoiceDate (datetime), UnitPrice (float), CustomerID (int), Country (str), Revenue (float)

Return ONLY a JSON object:
{"code": "your_python_code_here", "explanation": "one sentence"}

Rules: use pd/np/plt only. Store answer in `result`. Charts: assign figure to `fig`, do NOT call plt.show(). No file writes or imports."""

def ask_gpt_for_code(question, chat_history):
    messages = [{"role": "system", "content": SCHEMA_PROMPT}]
    for msg in chat_history[-4:]: messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=600, temperature=0)
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    parsed = ast.literal_eval(raw) if raw.startswith("{") else eval(raw)
    return parsed.get("code", ""), parsed.get("explanation", "")

def execute_code(code):
    local_vars = {"segments": segments, "cleaned": cleaned, "pd": pd, "np": np, "plt": plt, "result": None, "fig": None}
    try:
        exec(code, {}, local_vars)
        return local_vars.get("result"), local_vars.get("fig"), None
    except Exception:
        return None, None, traceback.format_exc()

def format_result(result):
    if result is None: return "No result returned."
    if isinstance(result, pd.DataFrame): return result.to_string(index=False)
    if isinstance(result, pd.Series): return result.to_string()
    return str(result)

def narrate_result(question, result_str, chat_history):
    messages = [{"role": "system", "content": (
        "You are a retail analytics assistant. Python ran a query on the user's data. "
        "Give a clear, concise, business-focused answer in 2-4 sentences. "
        "Include key numbers and one actionable insight. Do not mention Python or code."
    )}]
    for msg in chat_history[-4:]: messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": f"Question: {question}\n\nData result:\n{result_str}"})
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, max_tokens=300)
    return response.choices[0].message.content.strip()

seg_counts  = segments["Segment"].value_counts()
seg_revenue = segments.groupby("Segment")["Monetary"].sum()
total_rev   = cleaned["Revenue"].sum()
n_countries = cleaned["Country"].nunique()

# ══════════════════════════════════════════════════════════════════════════════
# WELCOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "welcome":

    st.markdown("<h1 style='text-align:center;'>🚀 RetailIQ</h1>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align:center; color:#94a3b8;'>AI-Powered Retail Decision System for Merchants</h3>", unsafe_allow_html=True)

    st.markdown("---")

    # Key Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Customers", f"{len(segments):,}")
    col2.metric("💷 Total Revenue (2010-2011) ", f"£{cleaned['Revenue'].sum():,.0f}")
    col3.metric("🌍 Active Countries", cleaned['Country'].nunique())

    st.markdown("---")

    # Clean Feature Highlights
    st.markdown("### ⚡ What you can do")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
- 📊 Segment customers  
- 🎯 Identify high-value users  
- ⚠️ Detect churn risk  
        """)

    with col2:
        st.markdown("""
- 🛍️ Recommend products  
- 📈 Track revenue trends  
- 🤖 Ask AI questions  
        """)

    st.markdown("---")

    st.markdown("<h4 style='text-align:center;'>Start analyzing your business data</h4>", unsafe_allow_html=True)

    if st.button("🚀 Enter Dashboard", use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()

    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Back to Welcome Page
    if st.button("🏠 Home"):
       st.session_state.page = "welcome"
       st.rerun()
    st.markdown("## 📊 RetailIQ AI")
    st.markdown('<p style="color:#64748b;font-size:0.8rem;margin-top:-8px;">Customer Intelligence Platform</p>', unsafe_allow_html=True)
    st.markdown("---")
    tab_choice = st.radio(
        "Navigation",
        ["🏠  Dashboard", "👤  Customer Intelligence", "💬  Data Chat",
         "🛍️  Product Explorer", "📈  Revenue Trends", "⚖️  About & Ethics"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown('<p style="color:#475569;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;">Live Stats</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#94a3b8;font-size:0.85rem;">👥 <b style="color:#f1f5f9;">{len(segments):,}</b> customers</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#94a3b8;font-size:0.85rem;">🧾 <b style="color:#f1f5f9;">{len(cleaned):,}</b> transactions</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#94a3b8;font-size:0.85rem;">💷 <b style="color:#f1f5f9;">£{total_rev:,.0f}</b> total revenue</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:#94a3b8;font-size:0.85rem;">🌍 <b style="color:#f1f5f9;">{n_countries}</b> countries</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p style="color:#334155;font-size:0.72rem;">UCI Online Retail · Dec 2010–Dec 2011<br>DSC 550 Capstone · UMass Dartmouth</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BUSINESS OVERVIEW (FINAL CLEAN VERSION)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "dashboard" and tab_choice == "🏠  Dashboard":

    st.markdown("## 📊 Business Overview")

    # ─────────────────────────────────────────────
    # CUSTOMER KPI CARDS
    # ─────────────────────────────────────────────
    total_customers = len(segments)
    high_value = seg_counts.get("High-Value Customers", 0)
    regular = seg_counts.get("Regular Customers", 0)
    low_value = seg_counts.get("Low-Value Customers", 0)

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(kpi_card("Total Customers", f"{total_customers:,}", "all customers"), unsafe_allow_html=True)
    c2.markdown(kpi_card("High Value Customers", f"{high_value}", "top spenders"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Regular Customers", f"{regular:,}", "active buyers"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Low Activity Customers", f"{low_value:,}", "less engaged"), unsafe_allow_html=True)

    st.markdown("")

    # ─────────────────────────────────────────────
    # REVENUE KPI CARDS
    # ─────────────────────────────────────────────
    monthly = cleaned.copy()
    monthly["InvoiceDate"] = pd.to_datetime(monthly["InvoiceDate"])
    monthly["Month"] = monthly["InvoiceDate"].dt.to_period("M")
    monthly_rev = monthly.groupby("Month")["Revenue"].sum().reset_index()

    best_month = monthly_rev.loc[monthly_rev["Revenue"].idxmax()]
    worst_month = monthly_rev.loc[monthly_rev["Revenue"].idxmin()]
    avg_monthly = monthly_rev["Revenue"].mean()
    total_revenue = cleaned["Revenue"].sum()

    r1, r2, r3, r4 = st.columns(4)

    r1.markdown(kpi_card("Total Revenue", f"£{total_revenue:,.0f}", "full period"), unsafe_allow_html=True)
    r2.markdown(kpi_card("Best Month", str(best_month["Month"]), "highest sales"), unsafe_allow_html=True)
    r3.markdown(kpi_card("Lowest Month", str(worst_month["Month"]), "lowest sales"), unsafe_allow_html=True)
    r4.markdown(kpi_card("Monthly Average", f"£{avg_monthly:,.0f}", "avg per month"), unsafe_allow_html=True)

    st.caption("Dataset: December 2010 – December 2011")

    st.markdown("---")

    # ─────────────────────────────────────────────
    # SIDE-BY-SIDE CHARTS
    # ─────────────────────────────────────────────
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    col1, col2 = st.columns(2)

    # CUSTOMER CHART
    with col1:
        st.markdown("### 👥 Customers")

        labels = ["Regular", "Low", "High"]
        values = [regular, low_value, high_value]

        fig1, ax1 = plt.subplots(figsize=(4,3))
        ax1.bar(labels, values, color=['#4F8BF9', '#f59e0b', '#22c55e'])

        ax1.set_title("Customer Groups", color='white')
        ax1.set_ylabel("Customers", color='white')

        ax1.tick_params(colors='white')
        fig1.patch.set_facecolor('#0f172a')
        ax1.set_facecolor('#0f172a')

        plt.grid(alpha=0.2)

        st.pyplot(fig1)

    # REVENUE CHART
    with col2:
        st.markdown("### 📈 Revenue")

        months = monthly_rev["Month"].astype(str)
        revenue = monthly_rev["Revenue"]

        fig2, ax2 = plt.subplots(figsize=(4,3))

        ax2.plot(months, revenue, marker='o', color='#4F8BF9')

        # Highlight peak
        peak_idx = revenue.idxmax()
        peak_value = revenue.max()
        peak_month = months[peak_idx]

        ax2.scatter(peak_month, peak_value, color='red', s=80)
        ax2.text(peak_month, peak_value, f"£{peak_value:,.0f}", fontsize=9, color='white')

        # Format Y-axis (£)
        ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"£{int(x):,}"))

        ax2.set_title("Monthly Revenue", color='white')
        ax2.set_ylabel("Revenue (£)", color='white')

        ax2.tick_params(colors='white')
        fig2.patch.set_facecolor('#0f172a')
        ax2.set_facecolor('#0f172a')

        plt.xticks(rotation=45)
        plt.grid(alpha=0.2)

        st.pyplot(fig2)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # BUSINESS TIPS
    # ─────────────────────────────────────────────
    st.markdown("### 💡 Tips to Improve Revenue")

    st.info("""
• Focus on high-value customers — they generate most revenue  
• Convert regular customers into high-value buyers  
• Re-engage low activity customers with offers  
• Plan promotions around high-performing months  
""")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "👤  Customer Intelligence":

    page_header(
        "Customer Intelligence",
        "Select a customer — view profile, predictions, recommendations, and insights"
    )

    st.markdown(
        '<span class="tip-chip">⭐ High-Value</span>'
        '<span class="tip-chip">🔵 Regular</span>'
        '<span class="tip-chip">⚪ Low-Value</span>',
        unsafe_allow_html=True
    )

    st.markdown("")

    # ─────────────────────────────────────────────
    # SMART CUSTOMER SELECTION + SEARCH
    # ─────────────────────────────────────────────
    st.markdown("### 🔍 Select or Search Customer")

    col1, col2 = st.columns(2)

    # LEFT → Dropdown selection
    with col1:
        segment_choice = st.selectbox(
            "Customer Group",
            ["High-Value Customers", "Regular Customers", "Low-Value Customers"]
        )

        filtered_customers = segments[segments["Segment"] == segment_choice]

        selected_customer = st.selectbox(
            "Select Customer",
            filtered_customers["CustomerID"].tolist()
        )

    # RIGHT → Search
    with col2:
        search_customer = st.text_input(
            "Search Customer ID",
            placeholder="e.g. 12415"
        )

    # FINAL CUSTOMER ID
    if search_customer:
        try:
            cid = int(search_customer.strip())
        except ValueError:
            st.error("Please enter a valid numeric Customer ID.")
            st.stop()
    else:
        cid = selected_customer

    # VALIDATION
    row = segments[segments["CustomerID"] == cid]
    if row.empty:
        st.markdown(f'<div class="info-box">❌ Customer <b>{cid}</b> not found.</div>', unsafe_allow_html=True)
        st.stop()

    row = row.iloc[0]

    recency = int(row["Recency"])
    frequency = int(row["Frequency"])
    monetary = float(row["Monetary"])
    segment = row["Segment"]

    # ─────────────────────────────────────────────
    # CUSTOMER PROFILE
    # ─────────────────────────────────────────────
    section("Customer Profile")

    p1, p2, p3, p4 = st.columns(4)

    seg_style = "kpi-gold" if "High" in segment else "kpi-blue" if "Regular" in segment else "kpi-slate"

    p1.markdown(kpi_card("Segment", segment.replace(" Customers",""), "", seg_style), unsafe_allow_html=True)
    p2.markdown(kpi_card("Last Purchase", f"{recency} days ago", "", "kpi-blue"), unsafe_allow_html=True)
    p3.markdown(kpi_card("Orders", f"{frequency}", "", "kpi-green"), unsafe_allow_html=True)
    p4.markdown(kpi_card("Total Spend", f"£{monetary:,.0f}", "", "kpi-gold"), unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # ML PREDICTION
    # ─────────────────────────────────────────────
    try:
        features = np.array([[recency, frequency]])
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        prob_pct = f"{max(proba)*100:.0f}%"

        if prediction == 1:
            st.markdown(
                f'<div class="pred-high">✅ High-value customer (Confidence: {prob_pct})</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="pred-low">ℹ️ Not a high-value customer (Confidence: {prob_pct})</div>',
                unsafe_allow_html=True
            )

    except Exception:
        prediction = 0

    st.markdown("")

    # ─────────────────────────────────────────────
    # PURCHASE HISTORY ONLY
    # ─────────────────────────────────────────────
    cust_products = cleaned[cleaned["CustomerID"] == cid]["Description"].dropna().unique().tolist()

    section("Purchase History")

    with st.expander(f"View {len(cust_products)} products purchased"):
     for p in cust_products[:30]:
        st.markdown(f"• {p}")

    if len(cust_products) > 30:
        st.caption(f"...and {len(cust_products)-30} more")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    # ─────────────────────────────────────────────
    # AI ASSISTANT
    # ─────────────────────────────────────────────
    section("🤖 AI Assistant")

    st.caption("Ask: What strategy should I use? · Is this customer inactive? · What should I recommend?")

    system_context = build_customer_context(cid, row, [], prediction)

    chat_key = f"chat_{cid}"

    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input(f"Ask about customer {cid}...")

    if user_input:
        st.session_state[chat_key].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        messages = [{"role": "system", "content": system_context}]
        messages += st.session_state[chat_key]

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        max_tokens=350
                    )
                    reply = response.choices[0].message.content.strip()
                    st.markdown(reply)
                    st.session_state[chat_key].append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"OpenAI error: {e}")

    if st.session_state.get(chat_key):
        if st.button("🗑️ Clear chat"):
            st.session_state[chat_key] = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA CHAT
# ══════════════════════════════════════════════════════════════════════════════
if tab_choice == "💬  Data Chat":
    page_header("AI Data Chat", "Ask questions in plain English — queries run live against your full dataset")
    with st.expander("💡 Example questions"):
        st.markdown("""
**Data questions:**
- *How many customers haven't bought in 90 days?*
- *Show me the top 10 products by revenue*
- *Which country has the highest revenue?*
- *Compare average spend between segments*
- *Plot monthly revenue as a chart*

**General questions:**
- *What is RFM analysis?*
- *How does customer segmentation work?*
- *What does lift mean in market basket analysis?*
        """)
    if "data_chat_v3" not in st.session_state: st.session_state["data_chat_v3"] = []
    for msg in st.session_state["data_chat_v3"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("dataframe") is not None: st.dataframe(msg["dataframe"], use_container_width=True, hide_index=True)
            if msg.get("chart") is not None: st.pyplot(msg["chart"])
    user_question = st.chat_input("Ask anything about the data...")
    if user_question:
        st.session_state["data_chat_v3"].append({"role": "user", "content": user_question})
        with st.chat_message("user"): st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Analysing..."):
                try:
                    if not is_data_question(user_question):
                        reply = answer_general_question(user_question, st.session_state["data_chat_v3"])
                        st.markdown(reply)
                        st.session_state["data_chat_v3"].append({"role": "assistant", "content": reply})
                    else:
                        code, _ = ask_gpt_for_code(user_question, st.session_state["data_chat_v3"])
                        result, fig, error = execute_code(code)
                        if error:
                            reply = answer_general_question(user_question, st.session_state["data_chat_v3"])
                            st.markdown(reply)
                            st.session_state["data_chat_v3"].append({"role": "assistant", "content": reply})
                        else:
                            result_str = format_result(result)
                            reply = narrate_result(user_question, result_str, st.session_state["data_chat_v3"])
                            st.markdown(reply)
                            display_df = None
                            if isinstance(result, pd.DataFrame) and not result.empty:
                                st.dataframe(result, use_container_width=True, hide_index=True); display_df = result
                            elif isinstance(result, pd.Series):
                                df_d = result.reset_index(); st.dataframe(df_d, use_container_width=True, hide_index=True); display_df = df_d
                            display_fig = None
                            if fig is not None: st.pyplot(fig); display_fig = fig; plt.close(fig)
                            st.session_state["data_chat_v3"].append({"role": "assistant", "content": reply, "dataframe": display_df, "chart": display_fig})
                except Exception as e:
                    err = f"Error: {e}"; st.error(err)
                    st.session_state["data_chat_v3"].append({"role": "assistant", "content": err})
    if st.session_state.get("data_chat_v3"):
        if st.button("🗑️ Clear conversation"): st.session_state["data_chat_v3"] = []; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRODUCT EXPLORER 
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "🛍️  Product Explorer":

    page_header(
        "Product Explorer",
        "Understand what products sell best and what customers usually buy together"
    )

    # ─────────────────────────────────────────────
    # PRODUCT ASSOCIATIONS 
    # ─────────────────────────────────────────────
    section("Products Often Bought Together")

    st.caption(
        "(Confidence -> Chance of buying together) - These are products that customers frequently purchase together.\n "
        "\nHigher value of 'Lift' means customers are much more likely to purchase those items together."
    )

    search = st.text_input(
        "🔍 Search for a product",
        placeholder="e.g. TEACUP",
        label_visibility="collapsed"
    )

    rules_df = recs_raw[["antecedents", "consequents", "confidence", "lift"]].copy()

    rules_df["confidence"] = (rules_df["confidence"] * 100).round(1).astype(str) + "%"
    rules_df["lift"] = rules_df["lift"].round(2)

    rules_df.columns = [
        "If customer buys →",
        "They also tend to buy",
        "Chance of buying together",
        "Lift"
    ]

    if search:
        mask = (
            rules_df["If customer buys →"].str.contains(search.upper(), na=False) |
            rules_df["They also tend to buy"].str.contains(search.upper(), na=False)
        )
        rules_df = rules_df[mask]

    st.dataframe(rules_df.head(30), use_container_width=True, hide_index=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # TOP PRODUCTS 
    # ─────────────────────────────────────────────
    section("Top 15 Products by Revenue")

    st.caption(
        "These are your highest revenue-generating products. Focus on these to maximize profit."
    )

    top_prods = (
        cleaned.groupby("Description")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    top_prods.columns = ["Product", "Revenue"]

    fig, ax = dark_fig(11, 5)

    ax.barh(
        top_prods["Product"][::-1],
        top_prods["Revenue"][::-1],
        color=BLUE,
        edgecolor="#0f172a",
        linewidth=0.8,
        alpha=0.9
    )

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k")
    )

    ax.set_xlabel("Total Revenue (£)", color="#64748b")
    ax.grid(axis="x", alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    st.pyplot(fig)
    plt.close()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # REVENUE BY COUNTRY 
    # ─────────────────────────────────────────────
    section("Revenue by Country")

    col_all, col_excl = st.columns(2)

    with col_all:
        st.caption("All countries (UK dominates)")

        cr = (
            cleaned.groupby("Country")["Revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig, ax = dark_fig(5.5, 4)

        ax.bar(cr["Country"], cr["Revenue"], color=PURPLE, edgecolor="#0f172a", alpha=0.9)

        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k")
        )

        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig)
        plt.close()

    with col_excl:
        st.caption("Excluding UK (international view)")

        cr_ex = (
            cleaned[cleaned["Country"] != "United Kingdom"]
            .groupby("Country")["Revenue"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        fig, ax = dark_fig(5.5, 4)

        ax.bar(cr_ex["Country"], cr_ex["Revenue"], color=CORAL, edgecolor="#0f172a", alpha=0.9)

        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k")
        )

        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — REVENUE TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "📈  Revenue Trends":
    page_header("Revenue Trends", "Monthly performance — Dec 2010 to Dec 2011")
    monthly = cleaned.set_index("InvoiceDate").resample("ME")["Revenue"].sum().reset_index()
    monthly.columns = ["Month", "Revenue"]
    monthly["MoM_Change"] = monthly["Revenue"].pct_change() * 100
    peak = monthly.loc[monthly["Revenue"].idxmax()]
    low  = monthly.loc[monthly["Revenue"].idxmin()]
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi_card("Total Revenue", f"£{monthly['Revenue'].sum():,.0f}", "Dec 2010–Dec 2011", "kpi-green"), unsafe_allow_html=True)
    k2.markdown(kpi_card("Best Month", peak["Month"].strftime("%b %Y"), f"£{peak['Revenue']:,.0f}", "kpi-gold"), unsafe_allow_html=True)
    k3.markdown(kpi_card("Lowest Month", low["Month"].strftime("%b %Y"), f"£{low['Revenue']:,.0f}", "kpi-slate"), unsafe_allow_html=True)
    k4.markdown(kpi_card("Monthly Average", f"£{monthly['Revenue'].mean():,.0f}", f"across {len(monthly)} months", "kpi-blue"), unsafe_allow_html=True)
    st.markdown("")
    section("Monthly Revenue")
    fig, ax = dark_fig(11, 4.5)
    ax.plot(monthly["Month"], monthly["Revenue"], marker="o", color=BLUE, linewidth=2.5, markersize=6, zorder=3)
    ax.fill_between(monthly["Month"], monthly["Revenue"], alpha=0.12, color=BLUE)
    ax.annotate(f"Peak: £{peak['Revenue']:,.0f}", xy=(peak["Month"], peak["Revenue"]),
                xytext=(0, 14), textcoords="offset points", ha="center", color=GOLD, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=GOLD, lw=0.8))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1000:.0f}k"))
    ax.tick_params(axis="x", rotation=30); ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    st.pyplot(fig); plt.close()
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    section("Month-over-Month Change (%)")
    fig, ax = dark_fig(11, 3)
    colors_mom = [GREEN if v >= 0 else CORAL for v in monthly["MoM_Change"].fillna(0)]
    ax.bar(monthly["Month"], monthly["MoM_Change"].fillna(0), color=colors_mom, edgecolor="#0f172a", width=20)
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.tick_params(axis="x", rotation=30); ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    st.pyplot(fig); plt.close()
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    section("Monthly Revenue Table")
    display = monthly.copy()
    display["Month"]      = display["Month"].dt.strftime("%B %Y")
    display["Revenue"]    = display["Revenue"].map(lambda x: f"£{x:,.2f}")
    display["MoM Change"] = monthly["MoM_Change"].map(lambda x: f"+{x:.1f}%" if x >= 0 else f"{x:.1f}%" if not pd.isna(x) else "—")
    st.dataframe(display[["Month","Revenue","MoM Change"]], use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ABOUT & ETHICS
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == "⚖️  About & Ethics":
    page_header("About & Ethical Considerations", "DSC 550 Data Science Practicum · UMass Dartmouth")
    col_about, col_ethics = st.columns([1, 1])
    with col_about:
        section("Project Overview")
        st.markdown("""
An end-to-end **customer analytics pipeline** on the UCI Online Retail dataset.

**Dataset**
- Source: UCI Machine Learning Repository
- Retailer: UK online gift store
- Period: Dec 2010 – Dec 2011
- Raw: 541,909 → Cleaned: 392,578 rows
- Unique customers: 4,331

**9-Notebook Pipeline**

| Notebook | Step |
|----------|------|
| 01 | Data Loading |
| 02 | Data Cleaning |
| 03 | Exploratory Analysis |
| 04 | RFM Feature Engineering |
| 05 | K-Means Segmentation |
| 06 | Apriori Recommendations |
| 07 | Model Training (5 models) |
| 08 | Model Evaluation |
| 09 | Business Integration |
        """)
        section("Model Performance")
        st.markdown("""
| Metric | Value |
|--------|-------|
| Algorithm | Logistic Regression |
| Accuracy | ~83% |
| AUC-ROC | ~0.90 |
| Features | Recency, Frequency |
| Target | Monetary > median |
        """)
        section("Technologies")
        st.markdown("""
pandas · numpy · matplotlib · seaborn  
sklearn · XGBoost · mlxtend · joblib  
Streamlit · OpenAI gpt-4o-mini
        """)
    with col_ethics:
        section("Ethical Considerations")
        st.markdown("""
**1. Data Privacy**
CustomerIDs are anonymised — no PII present. Original customers did not explicitly consent
to ML research use. GDPR-compliant consent notices required in production.

---
**2. Geographic Bias**
91% of data is from the United Kingdom. The model may not generalise well to international
customers. Country-specific models recommended for global deployment.

---
**3. Fairness — Segment Treatment**
'Low-Value' labels are based on past behaviour, not future potential. Permanently
deprioritising these customers creates a self-fulfilling exclusion. Quarterly segment
reviews are recommended to allow tier movement.

---
**4. Model Transparency**
Logistic Regression was chosen over higher-accuracy alternatives for its interpretability.
Coefficients directly explain predictions — essential for stakeholder trust and compliance.

---
**5. Data Security**
API keys stored in `.env`, not in code. No customer PII sent to external AI services.
All data remains local.

---
**6. Temporal Validity**
Model trained on 2011 data. Behaviour, preferences, and market conditions change.
Retraining on current data required before any production deployment.
        """)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    section("One-Line Summary")
    st.markdown("""
> *"An end-to-end customer analytics system using RFM segmentation, market basket analysis, and purchase prediction —
> packaged into a Streamlit app with an AI chat interface that lets business users query customer insights in plain English."*
    """)
