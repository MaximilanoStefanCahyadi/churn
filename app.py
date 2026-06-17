import os, pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                              average_precision_score, confusion_matrix)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Intelligence", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════════════════════
# THEME ENGINE
# ══════════════════════════════════════════════════════════════════════════════
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

THEMES = {
    "dark": {
        "bg":           "#0b0f1a",
        "bg_secondary": "#111827",
        "surface":      "rgba(255,255,255,0.04)",
        "surface_hover":"rgba(255,255,255,0.07)",
        "card_border":  "rgba(255,255,255,0.08)",
        "card_shadow":  "rgba(0,0,0,0.4)",
        "text":         "#f1f5f9",
        "text_secondary":"#94a3b8",
        "text_muted":   "#64748b",
        "divider":      "rgba(255,255,255,0.06)",
        "success":      "#10b981",
        "warning":      "#f59e0b",
        "danger":       "#ef4444",
        "accent_1":     "#6366f1",
        "accent_2":     "#8b5cf6",
        "badge_green_bg":  "rgba(16,185,129,0.15)",
        "badge_green_fg":  "#34d399",
        "badge_yellow_bg": "rgba(245,158,11,0.15)",
        "badge_yellow_fg": "#fbbf24",
        "badge_red_bg":    "rgba(239,68,68,0.15)",
        "badge_red_fg":    "#f87171",
        "input_bg":     "rgba(255,255,255,0.05)",
        "input_border": "rgba(255,255,255,0.1)",
        "tab_bg":       "rgba(255,255,255,0.03)",
        "tab_active_bg":"rgba(255,255,255,0.1)",
        "scrollbar_track": "#1a1f2e",
        "scrollbar_thumb": "#334155",
        # Matplotlib
        "fig_bg":       "#0b0f1a",
        "axes_bg":      "#111827",
        "axes_edge":    "#1e293b",
        "grid_color":   "#1e293b",
        "tick_color":   "#94a3b8",
        "label_color":  "#cbd5e1",
        "title_color":  "#f1f5f9",
    },
    "light": {
        "bg":           "#f8fafc",
        "bg_secondary": "#f1f5f9",
        "surface":      "rgba(255,255,255,0.75)",
        "surface_hover":"rgba(255,255,255,0.9)",
        "card_border":  "rgba(0,0,0,0.07)",
        "card_shadow":  "rgba(0,0,0,0.06)",
        "text":         "#0f172a",
        "text_secondary":"#475569",
        "text_muted":   "#94a3b8",
        "divider":      "rgba(0,0,0,0.06)",
        "success":      "#16a34a",
        "warning":      "#d97706",
        "danger":       "#dc2626",
        "accent_1":     "#6366f1",
        "accent_2":     "#8b5cf6",
        "badge_green_bg":  "#dcfce7",
        "badge_green_fg":  "#16a34a",
        "badge_yellow_bg": "#fef9c3",
        "badge_yellow_fg": "#a16207",
        "badge_red_bg":    "#fee2e2",
        "badge_red_fg":    "#dc2626",
        "input_bg":     "rgba(0,0,0,0.03)",
        "input_border": "rgba(0,0,0,0.1)",
        "tab_bg":       "rgba(0,0,0,0.03)",
        "tab_active_bg":"rgba(255,255,255,0.95)",
        "scrollbar_track": "#e2e8f0",
        "scrollbar_thumb": "#94a3b8",
        # Matplotlib
        "fig_bg":       "#f8fafc",
        "axes_bg":      "#ffffff",
        "axes_edge":    "#e2e8f0",
        "grid_color":   "#e2e8f0",
        "tick_color":   "#475569",
        "label_color":  "#334155",
        "title_color":  "#0f172a",
    },
}

T = THEMES[st.session_state.theme]

# Chart color palette (consistent across both themes)
CHART_COLORS = ["#6366f1", "#8b5cf6", "#a78bfa", "#14b8a6", "#f59e0b",
                "#ef4444", "#ec4899", "#06b6d4", "#84cc16", "#f97316"]
CHART_PALETTE = sns.color_palette(CHART_COLORS)
CHART_CMAP_SEQ = sns.color_palette(["#312e81", "#4338ca", "#6366f1", "#818cf8", "#c7d2fe"], as_cmap=True)

def apply_mpl_theme():
    """Apply matplotlib rcParams based on current theme."""
    plt.rcParams.update({
        'figure.facecolor':  T["fig_bg"],
        'axes.facecolor':    T["axes_bg"],
        'axes.edgecolor':    T["axes_edge"],
        'axes.labelcolor':   T["label_color"],
        'axes.titlecolor':   T["title_color"],
        'xtick.color':       T["tick_color"],
        'ytick.color':       T["tick_color"],
        'text.color':        T["label_color"],
        'grid.color':        T["grid_color"],
        'grid.alpha':        0.5,
        'axes.grid':         False,
        'figure.dpi':        100,
        'savefig.facecolor': T["fig_bg"],
        'font.family':       'sans-serif',
        'font.size':         10,
    })

apply_mpl_theme()


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {{
  --bg:           {T["bg"]};
  --bg2:          {T["bg_secondary"]};
  --surface:      {T["surface"]};
  --surface-h:    {T["surface_hover"]};
  --border:       {T["card_border"]};
  --shadow:       {T["card_shadow"]};
  --text:         {T["text"]};
  --text2:        {T["text_secondary"]};
  --text3:        {T["text_muted"]};
  --divider:      {T["divider"]};
  --accent1:      {T["accent_1"]};
  --accent2:      {T["accent_2"]};
  --success:      {T["success"]};
  --warning:      {T["warning"]};
  --danger:       {T["danger"]};
  --input-bg:     {T["input_bg"]};
  --input-brd:    {T["input_border"]};
}}

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif !important; }}

/* ── Hide Streamlit chrome ────────────────────────────── */
#MainMenu, footer {{ visibility: hidden; }}
header {{ background-color: transparent !important; }}

/* ── Main area ────────────────────────────────────────── */
.stApp, .main .block-container {{
  background-color: var(--bg) !important;
  color: var(--text) !important;
}}
.block-container {{ padding: 1rem 2.5rem 2rem 2.5rem !important; }}

/* ── Sidebar ──────────────────────────────────────────── */
section[data-testid="stSidebar"] {{
  background: linear-gradient(170deg, #070a12 0%, #111827 50%, #0f172a 100%) !important;
  border-right: 1px solid rgba(255,255,255,0.06);
}}
section[data-testid="stSidebar"] * {{
  color: #e2e8f0 !important;
}}
section[data-testid="stSidebar"] label {{
  color: #94a3b8 !important; font-size:12px !important; font-weight:500 !important;
}}
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stRadio {{
  color: #e2e8f0 !important;
}}
section[data-testid="stSidebar"] .stSelectbox > div > div {{
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.1) !important;
  border-radius: 8px !important;
}}
section[data-testid="stSidebar"] hr {{
  border-color: rgba(255,255,255,0.08) !important;
  margin: 0.75rem 0 !important;
}}

/* ── Glass card ───────────────────────────────────────── */
.glass-card {{
  background: var(--surface);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem;
  box-shadow: 0 4px 24px var(--shadow);
  animation: fadeSlideIn 0.5s ease-out;
}}
.glass-card-sm {{
  background: var(--surface);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.15rem 1.25rem;
  box-shadow: 0 2px 12px var(--shadow);
  animation: fadeSlideIn 0.45s ease-out;
}}

/* ── Animations ───────────────────────────────────────── */
@keyframes fadeSlideIn {{
  from {{ opacity: 0; transform: translateY(12px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes pulse-ring {{
  0%   {{ box-shadow: 0 0 0 0 rgba(99,102,241,0.4); }}
  70%  {{ box-shadow: 0 0 0 10px rgba(99,102,241,0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(99,102,241,0); }}
}}
@keyframes fadeIn {{
  from {{ opacity:0; }} to {{ opacity:1; }}
}}

/* ── Metric card ──────────────────────────────────────── */
.metric-card {{
  background: var(--surface);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.15rem 1.35rem;
  box-shadow: 0 2px 16px var(--shadow);
  height: 100%;
  animation: fadeSlideIn 0.5s ease-out;
}}
.metric-label {{
  font-size: 11px; font-weight: 600; color: var(--text3);
  letter-spacing: 0.07em; text-transform: uppercase; margin-bottom: 0.45rem;
}}
.metric-value {{
  font-size: 1.75rem; font-weight: 800; line-height: 1; color: var(--text);
}}
.metric-sub {{
  font-size: 11.5px; color: var(--text3); margin-top: 0.3rem;
}}

/* ── Section headers ──────────────────────────────────── */
.section-header {{
  font-size: 14px; font-weight: 700; letter-spacing: 0.03em;
  color: var(--text); padding-left: 12px;
  border-left: 3px solid var(--accent1);
  margin-bottom: 1rem; margin-top: 1.5rem;
}}

/* ── Prediction hero ──────────────────────────────────── */
.pred-hero {{
  background: var(--surface);
  backdrop-filter: blur(20px);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 2rem 2rem 1.75rem;
  box-shadow: 0 8px 32px var(--shadow);
  text-align: center;
  animation: fadeSlideIn 0.6s ease-out;
}}

/* ── Circular gauge ───────────────────────────────────── */
.gauge-wrap {{
  position: relative;
  width: 180px; height: 180px;
  margin: 0 auto 1.25rem;
}}
.gauge-ring {{
  width: 180px; height: 180px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  animation: fadeIn 0.8s ease-out;
}}
.gauge-inner {{
  width: 140px; height: 140px;
  border-radius: 50%;
  background: var(--bg);
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  box-shadow: inset 0 2px 8px rgba(0,0,0,0.2);
}}
.gauge-pct {{
  font-size: 2.5rem; font-weight: 900; line-height: 1;
}}
.gauge-label {{
  font-size: 11px; color: var(--text3); margin-top: 4px; font-weight: 500;
  text-transform: uppercase; letter-spacing: 0.05em;
}}

/* ── Badges ───────────────────────────────────────────── */
.badge {{
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 16px; border-radius: 99px;
  font-size: 13px; font-weight: 600;
}}
.badge-green  {{ background: {T["badge_green_bg"]};  color: {T["badge_green_fg"]}; }}
.badge-yellow {{ background: {T["badge_yellow_bg"]}; color: {T["badge_yellow_fg"]}; }}
.badge-red    {{ background: {T["badge_red_bg"]};    color: {T["badge_red_fg"]}; }}

/* ── Progress / confidence track ──────────────────────── */
.conf-track {{
  background: var(--input-bg); border-radius: 99px;
  height: 8px; overflow: hidden; margin: 0.4rem 0;
}}
.conf-fill {{
  height: 100%; border-radius: 99px;
  transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}}

/* ── Stay/Churn split bar ─────────────────────────────── */
.split-bar {{
  display: flex; border-radius: 99px; overflow: hidden;
  height: 28px; margin: 0.75rem 0;
  border: 1px solid var(--border);
}}
.split-stay {{
  background: var(--success); display: flex; align-items: center;
  justify-content: center; font-size: 12px; font-weight: 600;
  color: white; transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
}}
.split-churn {{
  background: var(--danger); display: flex; align-items: center;
  justify-content: center; font-size: 12px; font-weight: 600;
  color: white; transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
}}

/* ── Profile row ──────────────────────────────────────── */
.profile-row {{
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.5rem 0.75rem; border-radius: 8px;
  font-size: 13px; transition: background 0.2s;
}}
.profile-row:nth-child(odd) {{
  background: var(--surface);
}}
.profile-row:hover {{
  background: var(--surface-h);
}}
.profile-key {{
  color: var(--text3); font-weight: 500; display: flex; align-items: center; gap: 6px;
}}
.profile-val {{
  color: var(--text); font-weight: 600;
}}

/* ── Tabs ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
  gap: 4px; background: var(--tab-bg, {T["tab_bg"]});
  border-radius: 12px; padding: 4px;
  border: 1px solid var(--border);
}}
.stTabs [data-baseweb="tab"] {{
  border-radius: 8px; padding: 10px 24px;
  font-size: 13px; font-weight: 500; color: var(--text3) !important;
  background: transparent !important;
}}
.stTabs [aria-selected="true"] {{
  background: linear-gradient(135deg, var(--accent1), var(--accent2)) !important;
  color: white !important;
  box-shadow: 0 2px 12px rgba(99,102,241,0.3);
}}
.stTabs [data-baseweb="tab-highlight"] {{
  display: none;
}}
.stTabs [data-baseweb="tab-border"] {{
  display: none;
}}

/* ── Page title ───────────────────────────────────────── */
.page-title {{
  font-size: 32px; font-weight: 900;
  background: linear-gradient(135deg, {T["accent_1"]}, {T["accent_2"]}, #ec4899);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.2rem;
  animation: fadeSlideIn 0.5s ease-out;
}}
.page-sub {{
  font-size: 14px; color: var(--text3); margin-bottom: 1.5rem;
  animation: fadeSlideIn 0.6s ease-out;
}}

/* ── Gradient divider ─────────────────────────────────── */
.gradient-line {{
  height: 3px; border-radius: 2px; margin: 0.75rem 0 1.5rem;
  background: linear-gradient(90deg, {T["accent_1"]}, {T["accent_2"]}, transparent);
  animation: fadeIn 0.8s ease-out;
}}

/* ── Chart wrapper ────────────────────────────────────── */
.chart-wrap {{
  background: var(--surface);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1rem;
  box-shadow: 0 2px 16px var(--shadow);
  margin-bottom: 1rem;
  animation: fadeSlideIn 0.5s ease-out;
}}

/* ── Scrollbar ────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {T["scrollbar_track"]}; }}
::-webkit-scrollbar-thumb {{ background: {T["scrollbar_thumb"]}; border-radius: 3px; }}

/* ── Dataframe overrides ──────────────────────────────── */
.stDataFrame {{ animation: fadeSlideIn 0.4s ease-out; }}

/* ── Sidebar section label ────────────────────────────── */
.sidebar-section {{
  font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: #64748b !important;
  margin: 0.75rem 0 0.5rem; padding-left: 2px;
  display: flex; align-items: center; gap: 6px;
}}

/* ── Sidebar prediction chip ──────────────────────────── */
.sidebar-chip {{
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 12px;
  padding: 0.85rem 1rem;
  margin-top: 0.5rem;
  text-align: center;
}}
.sidebar-chip-pct {{
  font-size: 1.75rem; font-weight: 800; line-height: 1;
}}
.sidebar-chip-label {{
  font-size: 11px; color: #94a3b8; margin-top: 4px;
}}

/* ── Filter badge ─────────────────────────────────────── */
.filter-count {{
  font-size: 13px; color: var(--text3); margin-bottom: 0.75rem;
  animation: fadeIn 0.4s ease-out;
}}
.filter-count b {{
  color: var(--text); font-weight: 700;
}}
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = [
    'credit_score', 'age', 'tenure', 'balance', 'products_number',
    'credit_card', 'active_member', 'estimated_salary',
    'country_Germany', 'country_Spain', 'gender_Male'
]

FEATURE_LABELS = ['Credit score','Age','Tenure','Balance','Products',
                  'Credit card','Active member','Est. salary',
                  'Country: Germany','Country: Spain','Gender: Male']


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Loads both the Baseline and SMOTE models into a dictionary."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models = {}

    model_files = {
        "SMOTE Optimized": "smote.pkl",
        "Base Model": "base.pkl"
    }

    for m_name, f_name in model_files.items():
        for p in [os.path.join(base_dir, f_name), f_name]:
            if os.path.exists(p):
                with open(p, 'rb') as f:
                    models[m_name] = pickle.load(f)
                break
    return models

@st.cache_resource
def load_scaler():
    base = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(base, 'scaler.pkl'), 'scaler.pkl']:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                return pickle.load(f)
    raise FileNotFoundError("scaler.pkl not found — place it next to app.py")

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(base, 'churn.csv'), 'churn.csv']:
        if os.path.exists(p):
            df = pd.read_csv(p)
            if 'customer_id' in df.columns:
                df = df.drop(columns=['customer_id'])
            return df
    raise FileNotFoundError("churn.csv not found")

@st.cache_data
def prepare_test_set(_scaler):
    df = load_data()
    df_enc = pd.get_dummies(df, columns=['country','gender'], drop_first=True)

    for col in FEATURES:
        if col not in df_enc.columns:
            df_enc[col] = 0

    X = df_enc[FEATURES].astype(float)
    y = df_enc['churn']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_test_scaled = X_test.copy()
    cols_to_scale = ['credit_score', 'age', 'balance', 'estimated_salary']
    X_test_scaled[cols_to_scale] = _scaler.transform(X_test[cols_to_scale])

    return X_test_scaled, y_test


# Load everything globally
models_dict = load_models()
scaler = load_scaler()
df = load_data()
X_test_scaled, y_test = prepare_test_set(scaler)

def build_input_df(credit_score, age, tenure, balance, products,
                   credit_card, active, country, gender):
    row = {
        'credit_score'     : float(credit_score),
        'age'              : float(age),
        'tenure'           : float(tenure),
        'balance'          : float(balance),
        'products_number'  : float(products),
        'credit_card'      : float(credit_card),
        'active_member'    : float(active),
        'estimated_salary' : float(0),
        'country_Germany'  : float(country == 'Germany'),
        'country_Spain'    : float(country == 'Spain'),
        'gender_Male'      : float(gender == 'Male'),
    }
    return pd.DataFrame([row])[FEATURES]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Branding + Theme toggle ──
    st.markdown("""
    <div style="padding:1rem 0 0.5rem">
      <div style="font-size:22px;font-weight:800;color:#f8fafc;display:flex;align-items:center;gap:8px">
        🏦 <span style="background:linear-gradient(135deg,#6366f1,#8b5cf6,#ec4899);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        Churn Intelligence</span>
      </div>
      <div style="font-size:11px;color:#64748b;margin-top:4px;letter-spacing:0.03em">
        Gradient Boosting Dashboard
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:3px;border-radius:2px;background:linear-gradient(90deg,#6366f1,#8b5cf6,transparent);margin:0.25rem 0 0.75rem"></div>', unsafe_allow_html=True)

    # Theme toggle
    theme_label = "🌙 Dark Mode" if st.session_state.theme == "dark" else "☀️ Light Mode"
    theme_toggle = st.toggle(theme_label, value=(st.session_state.theme == "dark"), key="theme_toggle")
    new_theme = "dark" if theme_toggle else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Model Selection ──
    st.markdown('<div class="sidebar-section">⚙️ Model Engine</div>', unsafe_allow_html=True)
    selected_model_name = st.selectbox("Active Prediction Engine:", list(models_dict.keys()), label_visibility="collapsed")
    active_model = models_dict[selected_model_name]

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Demographics ──
    st.markdown('<div class="sidebar-section">👤 Demographics</div>', unsafe_allow_html=True)
    country  = st.selectbox("Country",  ['France', 'Germany', 'Spain'])
    gender   = st.selectbox("Gender",   ['Female', 'Male'])
    age      = st.slider("Age",          18, 92,  39)
    credit_score = st.slider("Credit score", 350, 850, 650)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Account Details ──
    st.markdown('<div class="sidebar-section">💰 Account Details</div>', unsafe_allow_html=True)
    tenure   = st.slider("Tenure (years)", 0, 10, 5)
    balance  = st.slider("Balance ($)", 0, 250000, 76000, step=500)
    salary   = st.slider("Estimated salary ($)", 0, 200000, 100000, step=500)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Behavior ──
    st.markdown('<div class="sidebar-section">📋 Behavior</div>', unsafe_allow_html=True)
    products    = st.select_slider("Number of products", [1, 2, 3, 4], value=1)
    credit_card = st.radio("Has credit card?", ["Yes", "No"], horizontal=True)
    active      = st.radio("Active member?",   ["Yes", "No"], horizontal=True)

    cc_val = 1 if credit_card == "Yes" else 0
    am_val = 1 if active      == "Yes" else 0

    # Build and scale
    input_df = build_input_df(credit_score, age, tenure, balance,
                              products, cc_val, am_val, country, gender)
    input_df['estimated_salary'] = float(salary)

    cols_to_scale = ['credit_score', 'age', 'balance', 'estimated_salary']
    input_scaled = input_df.copy()
    input_scaled[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    churn_prob = float(active_model.predict_proba(input_scaled)[0][1])
    stay_prob  = 1 - churn_prob
    confidence = abs(churn_prob - 0.5) * 2

    if churn_prob < 0.30:
        verdict, badge_cls, bar_color = "Low Risk", "badge-green", T["success"]
    elif churn_prob < 0.55:
        verdict, badge_cls, bar_color = "At Risk", "badge-yellow", T["warning"]
    else:
        verdict, badge_cls, bar_color = "High Risk", "badge-red", T["danger"]

    conf_label = "High"   if confidence >= 0.7 else \
                 "Medium" if confidence >= 0.4 else "Low"
    conf_color = T["success"] if confidence >= 0.7 else \
                 T["warning"] if confidence >= 0.4 else T["danger"]

    # ── Sidebar prediction chip ──
    st.markdown("<hr>", unsafe_allow_html=True)
    churn_pct_sb = round(churn_prob * 100)
    st.markdown(f"""
    <div class="sidebar-chip">
      <span class="badge {badge_cls}" style="margin-bottom:8px">{verdict}</span>
      <div class="sidebar-chip-pct" style="color:{bar_color}">{churn_pct_sb}%</div>
      <div class="sidebar-chip-label">churn probability</div>
    </div>
    """, unsafe_allow_html=True)


# Get test predictions for the currently active model
proba_test_active = active_model.predict_proba(X_test_scaled)[:,1]


# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-title">Bank Customer Churn Intelligence</div>
<div class="page-sub">Predict churn risk and explore data-driven insights across 10,000 customers.</div>
<div class="gradient-line"></div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯 Predictor", "📊 Exploratory Analysis", "📋 Data Explorer"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Predictor
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    stay_pct  = round(stay_prob  * 100)
    churn_pct = round(churn_prob * 100)

    col_pred, col_detail = st.columns([1, 1.4], gap="large")

    # ── Left: Hero prediction card ──
    with col_pred:
        # Circular gauge with conic-gradient
        gauge_angle = churn_pct * 3.6  # percentage to degrees
        track_color = "rgba(255,255,255,0.06)" if st.session_state.theme == "dark" else "rgba(0,0,0,0.06)"

        st.markdown(f"""
        <div class="pred-hero">
          <span class="badge {badge_cls}" style="margin-bottom:1rem">{verdict}</span>

          <div class="gauge-wrap">
            <div class="gauge-ring" style="background: conic-gradient(
              {bar_color} 0deg, {bar_color} {gauge_angle}deg,
              {track_color} {gauge_angle}deg, {track_color} 360deg
            );">
              <div class="gauge-inner">
                <div class="gauge-pct" style="color:{bar_color}">{churn_pct}%</div>
                <div class="gauge-label">Churn Risk</div>
              </div>
            </div>
          </div>

          <div style="margin-top:0.75rem">
            <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text3);margin-bottom:4px">
              <span>Low risk</span><span>High risk</span>
            </div>
            <div class="conf-track">
              <div class="conf-fill" style="width:{churn_pct}%;background:{bar_color};"></div>
            </div>
          </div>

          <div class="split-bar" style="margin-top:1rem;">
            <div class="split-stay"  style="width:{stay_pct}%;">{stay_pct}% stay</div>
            <div class="split-churn" style="width:{churn_pct}%;">{churn_pct}% churn</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Right: Stats + Profile ──
    with col_detail:
        k1, k2, k3 = st.columns(3)
        stats = [
            (k1, "📊", "Risk Score",      f"{churn_pct}%", bar_color),
            (k2, "🛡️", "Retention",      f"{stay_pct}%",  T["success"]),
            (k3, "🎯", "Confidence",      conf_label,      conf_color),
        ]
        for col, icon, label, val, color in stats:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{icon} {label}</div>
                  <div class="metric-value" style="color:{color};font-size:1.6rem">{val}</div>
                </div>""", unsafe_allow_html=True)

        # Confidence detail card
        st.markdown(f"""
        <div class="glass-card-sm" style="margin-top:1rem">
          <div class="metric-label">🔬 Model Confidence</div>
          <div style="display:flex;align-items:baseline;gap:8px;">
            <div class="metric-value" style="color:{conf_color};font-size:1.4rem">{round(confidence*100)}%</div>
            <span style="font-size:12px;color:var(--text3);font-weight:500">{conf_label}</span>
          </div>
          <div class="conf-track" style="margin-top:0.5rem">
            <div class="conf-fill" style="width:{round(confidence*100)}%;background:{conf_color};"></div>
          </div>
          <div class="metric-sub">
            {"High confidence — prediction is reliable."       if conf_label=="High"   else
             "Moderate confidence — review manually."          if conf_label=="Medium" else
             "Low confidence — near decision boundary."}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Customer Profile
        st.markdown('<div class="section-header" style="margin-top:1.25rem">Customer Profile</div>', unsafe_allow_html=True)

        fields = [
            ("🌍", "Country", country),     ("👤", "Gender", gender),
            ("🎂", "Age", age),             ("💳", "Credit Score", credit_score),
            ("📅", "Tenure", f"{tenure} yrs"), ("💰", "Balance", f"${balance:,}"),
            ("📦", "Products", products),    ("💵", "Est. Salary", f"${salary:,}"),
            ("🏧", "Credit Card", credit_card), ("⚡", "Active", active),
        ]

        profile_html = '<div class="glass-card-sm" style="padding:0.5rem 0.75rem">'
        for icon, k, v in fields:
            profile_html += f"""
            <div class="profile-row">
              <span class="profile-key">{icon} {k}</span>
              <span class="profile-val">{v}</span>
            </div>"""
        profile_html += '</div>'
        st.markdown(profile_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Exploratory Data Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    apply_mpl_theme()

    # --- ROW 1: Churn Dist & Products ---
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.markdown('<div class="section-header" style="margin-top:0">1 · Churn Distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        palette1 = [CHART_COLORS[0], CHART_COLORS[5]]
        sns.countplot(data=df, x='churn', palette=palette1, ax=ax1, edgecolor='none')
        ax1.set_xlabel('Churn (0 = Stay, 1 = Leave)', fontsize=10, color=T["label_color"])
        ax1.set_ylabel('Count', fontsize=10, color=T["label_color"])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig1)
        st.markdown('</div>', unsafe_allow_html=True)

    with r1c2:
        st.markdown('<div class="section-header" style="margin-top:0">2 · Churn Rate by Products</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(5, 3.5))
        product_churn = df.groupby('products_number')['churn'].mean().reset_index()
        sns.barplot(data=product_churn, x='products_number', y='churn', palette=CHART_COLORS[:4], ax=ax4, edgecolor='none')
        for p in ax4.patches:
            ax4.annotate(f'{p.get_height():.1%}',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9, color=T["label_color"],
                         xytext=(0, 4), textcoords='offset points')
        ax4.set_xlabel('Number of Products', fontsize=10, color=T["label_color"])
        ax4.set_ylabel('Churn Rate', fontsize=10, color=T["label_color"])
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig4)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ROW 2: Demographics & Numerical Distributions ---
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.markdown('<div class="section-header">3 · Churn by Demographics</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        fig3, axes3 = plt.subplots(1, 2, figsize=(7, 4))
        pal_demo = [CHART_COLORS[0], CHART_COLORS[5]]
        sns.countplot(data=df, x='country', hue='churn', palette=pal_demo, ax=axes3[0], edgecolor='none')
        axes3[0].set_title('By Country', fontsize=11, color=T["title_color"])
        axes3[0].set_xlabel('')
        axes3[0].spines['top'].set_visible(False)
        axes3[0].spines['right'].set_visible(False)
        if axes3[0].get_legend():
            axes3[0].get_legend().get_frame().set_facecolor(T["axes_bg"])
            for text in axes3[0].get_legend().get_texts():
                text.set_color(T["label_color"])

        sns.countplot(data=df, x='gender', hue='churn', palette=pal_demo, ax=axes3[1], edgecolor='none')
        axes3[1].set_title('By Gender', fontsize=11, color=T["title_color"])
        axes3[1].set_xlabel('')
        axes3[1].spines['top'].set_visible(False)
        axes3[1].spines['right'].set_visible(False)
        if axes3[1].get_legend():
            axes3[1].get_legend().get_frame().set_facecolor(T["axes_bg"])
            for text in axes3[1].get_legend().get_texts():
                text.set_color(T["label_color"])
        plt.tight_layout()
        st.pyplot(fig3)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2c2:
        st.markdown('<div class="section-header">4 · Numerical Feature Distributions</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        num_cols = ['credit_score', 'age', 'balance', 'estimated_salary']
        fig2, axes2 = plt.subplots(2, 2, figsize=(7, 4))
        for i, col in enumerate(num_cols):
            row_idx, col_idx = i // 2, i % 2
            sns.histplot(data=df, x=col, kde=True, ax=axes2[row_idx, col_idx],
                         color=CHART_COLORS[i], bins=20, edgecolor='none', alpha=0.8)
            axes2[row_idx, col_idx].set_ylabel('')
            axes2[row_idx, col_idx].set_xlabel(col.replace('_', ' ').title(), fontsize=9, color=T["label_color"])
            axes2[row_idx, col_idx].spines['top'].set_visible(False)
            axes2[row_idx, col_idx].spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ROW 3: Correlation Heatmap & Feature Importance ---
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        st.markdown('<div class="section-header">5 · Correlation Heatmap</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        df_encoded = pd.get_dummies(df, columns=['country', 'gender'], drop_first=True)
        hm_cmap = 'RdBu_r' if st.session_state.theme == "light" else 'coolwarm'
        sns.heatmap(df_encoded.corr(), annot=True, cmap=hm_cmap, fmt='.2f',
                    linewidths=0.5, ax=ax5, annot_kws={"size": 7, "color": T["label_color"]},
                    cbar_kws={"shrink": 0.8})
        ax5.tick_params(labelsize=8, colors=T["tick_color"])
        cbar = ax5.collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(colors=T["tick_color"])
        plt.tight_layout()
        st.pyplot(fig5)
        st.markdown('</div>', unsafe_allow_html=True)

    with r3c2:
        st.markdown(f'<div class="section-header">6 · Feature Importance ({selected_model_name})</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        fig8, ax8 = plt.subplots(figsize=(6, 5))
        fi = pd.Series(active_model.feature_importances_, index=FEATURE_LABELS).sort_values(ascending=True)
        colors_fi = sns.color_palette([CHART_COLORS[0], CHART_COLORS[1], CHART_COLORS[2]], n_colors=len(fi))
        bars = ax8.barh(fi.index, fi.values, color=[CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(fi))],
                        edgecolor='none', height=0.65)
        ax8.set_xlabel('Importance Score', fontsize=10, color=T["label_color"])
        ax8.set_ylabel('')
        ax8.tick_params(labelsize=9, colors=T["tick_color"])
        ax8.spines['top'].set_visible(False)
        ax8.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig8)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- ROW 4: ROC and PR Comparison ---
    st.markdown('<div class="section-header">7 · Model Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    fig6, axes6 = plt.subplots(1, 2, figsize=(12, 4.5))
    line_colors = {'SMOTE Optimized': CHART_COLORS[4], 'Base Model': CHART_COLORS[0]}

    for m_name, m_obj in models_dict.items():
        m_proba = m_obj.predict_proba(X_test_scaled)[:,1]
        m_auc = roc_auc_score(y_test, m_proba)
        m_ap  = average_precision_score(y_test, m_proba)
        fpr, tpr, _ = roc_curve(y_test, m_proba)
        prec, rec, _ = precision_recall_curve(y_test, m_proba)

        lw = 2.5 if m_name == selected_model_name else 1.5
        alpha = 1.0 if m_name == selected_model_name else 0.45
        color = line_colors.get(m_name, CHART_COLORS[3])

        axes6[0].plot(fpr, tpr, color=color, lw=lw, alpha=alpha, label=f'{m_name} (AUC = {m_auc:.3f})')
        axes6[1].plot(rec, prec, color=color, lw=lw, alpha=alpha, label=f'{m_name} (AP = {m_ap:.3f})')

    axes6[0].plot([0, 1], [0, 1], color=T["text_muted"], lw=1.5, linestyle='--', alpha=0.5)
    axes6[0].set_xlabel('False Positive Rate', fontsize=10, color=T["label_color"])
    axes6[0].set_ylabel('True Positive Rate', fontsize=10, color=T["label_color"])
    axes6[0].set_title('ROC Curve Comparison', fontsize=12, color=T["title_color"], fontweight='bold')
    leg0 = axes6[0].legend(loc="lower right", fontsize=9, facecolor=T["axes_bg"], edgecolor=T["axes_edge"])
    for text in leg0.get_texts():
        text.set_color(T["label_color"])
    axes6[0].spines['top'].set_visible(False)
    axes6[0].spines['right'].set_visible(False)

    axes6[1].set_xlabel('Recall', fontsize=10, color=T["label_color"])
    axes6[1].set_ylabel('Precision', fontsize=10, color=T["label_color"])
    axes6[1].set_title('Precision-Recall Curve Comparison', fontsize=12, color=T["title_color"], fontweight='bold')
    leg1 = axes6[1].legend(loc="lower left", fontsize=9, facecolor=T["axes_bg"], edgecolor=T["axes_edge"])
    for text in leg1.get_texts():
        text.set_color(T["label_color"])
    axes6[1].spines['top'].set_visible(False)
    axes6[1].spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig6)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- ROW 5: Confusion Matrix Comparison ---
    st.markdown('<div class="section-header">8 · Confusion Matrix Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    fig7, axes7 = plt.subplots(1, 2, figsize=(12, 4.5))

    cm_cmaps = {'SMOTE Optimized': 'Purples', 'Base Model': 'Blues'}
    for i, (m_name, m_obj) in enumerate(models_dict.items()):
        m_preds = m_obj.predict(X_test_scaled)
        cm = confusion_matrix(y_test, m_preds)

        is_active = (m_name == selected_model_name)
        cmap = cm_cmaps.get(m_name, 'Greens')
        title_weight = 'bold' if is_active else 'normal'

        annot_color = T["label_color"]
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes7[i],
                    xticklabels=['Predicted Stay', 'Predicted Churn'],
                    yticklabels=['Actual Stay', 'Actual Churn'],
                    linewidths=2, linecolor=T["axes_bg"],
                    annot_kws={"color": annot_color, "fontsize": 12, "fontweight": "bold"})

        axes7[i].set_title(f'{m_name}', fontweight=title_weight, fontsize=12,
                           pad=12, color=T["title_color"])
        axes7[i].tick_params(colors=T["tick_color"], labelsize=9)
        # Fix colorbar tick colors
        cbar = axes7[i].collections[0].colorbar
        if cbar:
            cbar.ax.tick_params(colors=T["tick_color"])

    plt.tight_layout()
    st.pyplot(fig7)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header" style="margin-top:0.5rem">Full Dataset — 10,000 Customers</div>', unsafe_allow_html=True)

    # Filter bar
    st.markdown('<div class="glass-card-sm" style="margin-bottom:1rem">', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        filter_churn   = st.selectbox("Churn status", ["All", "Churned", "Stayed"], key="explorer_churn")
    with fc2:
        filter_country = st.selectbox("Country", ["All"] + sorted(df['country'].unique()), key="explorer_country")
    with fc3:
        age_range = st.slider("Age range", 18, 92, (18, 92), key="explorer_age")
    st.markdown('</div>', unsafe_allow_html=True)

    dff = df.copy()
    if filter_churn   == "Churned": dff = dff[dff['churn'] == 1]
    if filter_churn   == "Stayed":  dff = dff[dff['churn'] == 0]
    if filter_country != "All":     dff = dff[dff['country'] == filter_country]
    dff = dff[(dff['age'] >= age_range[0]) & (dff['age'] <= age_range[1])]

    st.markdown(f"""
    <div class="filter-count">
      Showing <b>{len(dff):,}</b> of <b>{len(df):,}</b> customers
    </div>""", unsafe_allow_html=True)

    disp = dff.reset_index(drop=True).copy()
    disp['churn']            = disp['churn'].map({0:'✅ Stay', 1:'🔴 Churn'})
    disp['balance']          = disp['balance'].apply(lambda x: f"${x:,.0f}")
    disp['estimated_salary'] = disp['estimated_salary'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(disp, use_container_width=True, height=420)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header" style="margin-top:0">Summary Statistics (filtered)</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-sm">', unsafe_allow_html=True)
    desc = dff[['credit_score','age','tenure','balance','estimated_salary']].describe().round(1).T[['mean','std','min','50%','max']]
    desc.columns = ['Mean','Std','Min','Median','Max']
    st.dataframe(desc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)