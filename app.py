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

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; }

section[data-testid="stSidebar"] {
  background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
  border-right: 1px solid #334155;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size:12px !important; }
section[data-testid="stSidebar"] .stSelectbox>div>div {
  background:#1e293b; border:1px solid #334155; color:#e2e8f0;
}

.metric-card {
  background:white; border-radius:14px; padding:1.25rem 1.5rem;
  box-shadow:0 1px 3px rgba(0,0,0,.07),0 4px 16px rgba(0,0,0,.04);
  border:1px solid #f1f5f9; height:100%;
}
.metric-label { font-size:12px; font-weight:500; color:#94a3b8;
  letter-spacing:.05em; text-transform:uppercase; margin-bottom:.4rem; }
.metric-value { font-size:2rem; font-weight:700; line-height:1; color:#0f172a; }
.metric-sub   { font-size:12px; color:#64748b; margin-top:.3rem; }

.section-header {
  font-size:13px; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
  color:#64748b; border-bottom:1px solid #e2e8f0; padding-bottom:.5rem; margin-bottom:1rem;
}

.pred-card {
  background:white; border-radius:16px; padding:2rem;
  box-shadow:0 2px 8px rgba(0,0,0,.08); border:1px solid #f1f5f9; text-align:center;
}
.pred-prob  { font-size:4rem; font-weight:800; line-height:1; }
.pred-label { font-size:13px; color:#64748b; margin-top:.25rem; }

.badge { display:inline-block; padding:4px 14px; border-radius:99px;
  font-size:13px; font-weight:600; margin-bottom:1rem; }
.badge-green  { background:#dcfce7; color:#16a34a; }
.badge-yellow { background:#fef9c3; color:#ca8a04; }
.badge-red    { background:#fee2e2; color:#dc2626; }

.conf-track { background:#f1f5f9; border-radius:99px; height:10px;
  overflow:hidden; margin:.5rem 0; }
.conf-fill  { height:100%; border-radius:99px; }

.split-bar  { display:flex; border-radius:99px; overflow:hidden;
  height:28px; margin:.75rem 0; border:1px solid #e2e8f0; }
.split-stay  { background:#22c55e; display:flex; align-items:center;
  justify-content:center; font-size:12px; font-weight:600; color:white; }
.split-churn { background:#ef4444; display:flex; align-items:center;
  justify-content:center; font-size:12px; font-weight:600; color:white; }

.factor-row { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
.factor-name { font-size:12px; color:#475569; width:130px; flex-shrink:0; }
.factor-bar-wrap { flex:1; background:#f1f5f9; border-radius:4px; height:8px; }
.factor-bar { height:8px; border-radius:4px; background:#3b82f6; }
.factor-pct { font-size:11px; color:#94a3b8; width:36px; text-align:right; }

.stTabs [data-baseweb="tab-list"] {
  gap:4px; background:#f8fafc; border-radius:10px;
  padding:4px; border:1px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"] {
  border-radius:7px; padding:8px 20px; font-size:13px; font-weight:500; color:#64748b;
}
.stTabs [aria-selected="true"] {
  background:white !important; color:#0f172a !important;
  box-shadow:0 1px 3px rgba(0,0,0,.1);
}
.page-title { font-size:28px; font-weight:700; color:#0f172a; margin-bottom:.25rem; }
.page-sub   { font-size:14px; color:#64748b; margin-bottom:1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
FEATURES = ['credit_score','age','tenure','balance','products_number',
            'credit_card','active_member','estimated_salary',
            'country_Germany','country_Spain','gender_Male']
FEATURE_LABELS = ['Credit score','Age','Tenure','Balance','Products',
                  'Credit card','Active member','Est. salary',
                  'Country: Germany','Country: Spain','Gender: Male']
COLOR_SEQ = ['#3b82f6','#22c55e','#ef4444','#f59e0b','#8b5cf6','#ec4899']

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(base, 'model.pkl'), 'model.pkl']:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                return pickle.load(f)
    raise FileNotFoundError("model.pkl not found — place it next to app.py")

@st.cache_resource
def load_scaler():
    base = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(base, 'scaler.pkl'), 'scaler.pkl']:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                return pickle.load(f)
    raise FileNotFoundError("scaler.pkl not found — place it next to app.py")

scaler = load_scaler()

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
def prepare_test_set(_model, _scaler): # Pass the scaler in here
    df = load_data()
    df_enc = pd.get_dummies(df, columns=['country','gender'], drop_first=True)
    
    for col in FEATURES:
        if col not in df_enc.columns:
            df_enc[col] = 0
            
    X = df_enc[FEATURES].astype(float)
    y = df_enc['churn']
    
    # Split the data
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- ADD THIS: Scale the test set before predicting! ---
    X_test_scaled = X_test.copy()
    cols_to_scale = ['credit_score', 'age', 'balance', 'estimated_salary'] # Use your exact names
    X_test_scaled[cols_to_scale] = _scaler.transform(X_test[cols_to_scale])
    # -------------------------------------------------------

    # Predict using the SCALED data
    proba = _model.predict_proba(X_test_scaled)[:,1]
    preds = (_model.predict(X_test_scaled)).astype(int)
    
    return X_test, y_test, proba, preds

# And update the loader line below it:


model = load_model()
df = load_data()
X_test, y_test, proba_test, preds_test = prepare_test_set(model, scaler)

def build_input_df(credit_score, age, tenure, balance, products,
                   credit_card, active, country, gender):
    row = {
        'credit_score'    : float(credit_score),
        'age'             : float(age),
        'tenure'          : float(tenure),
        'balance'         : float(balance),
        'products_number' : float(products),
        'credit_card'     : float(credit_card),
        'active_member'   : float(active),
        'estimated_salary': float(0),
        'country_Germany' : float(country == 'Germany'),
        'country_Spain'   : float(country == 'Spain'),
        'gender_Male'     : float(gender == 'Male'),
    }
    return pd.DataFrame([row])[FEATURES]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.25rem 0 1rem">
      <div style="font-size:20px;font-weight:700;color:#f8fafc">🏦 Churn Intelligence</div>
      <div style="font-size:12px;color:#64748b;margin-top:4px">Gradient Boosting · AUC 0.859</div>
    </div>
    <hr style="border-color:#334155;margin-bottom:1rem">
    <div style="font-size:11px;font-weight:600;letter-spacing:.08em;
         text-transform:uppercase;color:#64748b;margin-bottom:.75rem">
      Customer Profile
    </div>
    """, unsafe_allow_html=True)

    country  = st.selectbox("Country",  ['France', 'Germany', 'Spain'])
    gender   = st.selectbox("Gender",   ['Female', 'Male'])
    age      = st.slider("Age",          18, 92,  39)
    credit_score = st.slider("Credit score", 350, 850, 650)
    tenure   = st.slider("Tenure (years)", 0, 10, 5)
    balance  = st.slider("Balance ($)", 0, 250000, 76000, step=500)
    salary   = st.slider("Estimated salary ($)", 0, 200000, 100000, step=500)

    st.markdown("<hr style='border-color:#334155;margin:.75rem 0'>", unsafe_allow_html=True)
    st.markdown("""<div style="font-size:11px;font-weight:600;letter-spacing:.08em;
        text-transform:uppercase;color:#64748b;margin-bottom:.75rem">Behaviour</div>""",
        unsafe_allow_html=True)

    products    = st.select_slider("Number of products", [1, 2, 3, 4], value=1)
    credit_card = st.radio("Has credit card?", ["Yes", "No"], horizontal=True)
    active      = st.radio("Active member?",   ["Yes", "No"], horizontal=True)

    cc_val = 1 if credit_card == "Yes" else 0
    am_val = 1 if active      == "Yes" else 0

    # 1. Build your full 11-column dataframe
    input_df = build_input_df(credit_score, age, tenure, balance,
                              products, cc_val, am_val, country, gender)
    input_df['estimated_salary'] = float(salary)

    # 2. Define the exact columns the scaler expects
    cols_to_scale = ['credit_score', 'age', 'balance', 'estimated_salary']

    # 3. Create a copy of the dataframe so we don't mess up the original
    input_scaled = input_df.copy()

    # 4. Scale ONLY those four specific columns!
    input_scaled[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    # 5. Predict using the scaled dataframe
    churn_prob = float(model.predict_proba(input_scaled)[0][1])
    stay_prob  = 1 - churn_prob
    confidence = abs(churn_prob - 0.5) * 2

    if churn_prob < 0.30:
        verdict, badge_cls, bar_color = "Not Churning", "badge-green",  "#22c55e"
    elif churn_prob < 0.55:
        verdict, badge_cls, bar_color = "At Risk",      "badge-yellow", "#f59e0b"
    else:
        verdict, badge_cls, bar_color = "Likely Churning", "badge-red", "#ef4444"

    conf_label = "High"   if confidence >= 0.7 else \
                 "Medium" if confidence >= 0.4 else "Low"
    conf_color = "#22c55e" if confidence >= 0.7 else \
                 "#f59e0b" if confidence >= 0.4 else "#ef4444"

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-title">Bank Customer Churn Intelligence</div>
<div class="page-sub">Predict churn risk and explore model insights across 10,000 customers.</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎯  Predictor", "📊  Analytics", "📋  Data Explorer"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Predictor
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_pred, col_detail = st.columns([1, 1.6], gap="large")

    with col_pred:
        stay_pct  = round(stay_prob  * 100)
        churn_pct = round(churn_prob * 100)

        st.markdown(f"""
        <div class="pred-card">
          <span class="badge {badge_cls}">{verdict}</span>
          <div class="pred-prob" style="color:{bar_color}">{churn_pct}%</div>
          <div class="pred-label">churn probability</div>
          <div style="margin-top:1.5rem;">
            <div style="display:flex;justify-content:space-between;
                 font-size:12px;color:#94a3b8;margin-bottom:4px;">
              <span>Low risk</span><span>High risk</span>
            </div>
            <div class="conf-track">
              <div class="conf-fill"
                   style="width:{churn_pct}%;background:{bar_color};"></div>
            </div>
          </div>
          <div class="split-bar" style="margin-top:1rem;">
            <div class="split-stay"  style="width:{stay_pct}%;">{stay_pct}% stay</div>
            <div class="split-churn" style="width:{churn_pct}%;">{churn_pct}% churn</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Model Confidence</div>
          <div style="display:flex;align-items:baseline;gap:8px;">
            <div class="metric-value" style="color:{conf_color}">{round(confidence*100)}%</div>
            <span style="font-size:13px;color:#64748b;font-weight:500">{conf_label}</span>
          </div>
          <div class="conf-track" style="margin-top:.5rem;">
            <div class="conf-fill"
                 style="width:{round(confidence*100)}%;background:{conf_color};"></div>
          </div>
          <div class="metric-sub">
            {"High confidence — prediction is reliable."       if conf_label=="High"   else
             "Moderate confidence — review manually."          if conf_label=="Medium" else
             "Low confidence — near decision boundary."}
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_detail:
        k1, k2, k3 = st.columns(3)
        for col, label, val, color in [
            (k1, "Stay Probability",  f"{stay_pct}%",  "#22c55e"),
            (k2, "Churn Probability", f"{churn_pct}%", "#ef4444"),
            (k3, "Confidence",        conf_label,       conf_color),
        ]:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value" style="color:{color};font-size:1.5rem">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Top Risk Factors</div>',
                    unsafe_allow_html=True)

        fi_pairs = sorted(zip(FEATURE_LABELS, model.feature_importances_),
                          key=lambda x: -x[1])
        fi_html = ""
        for label, imp in fi_pairs[:7]:
            bar_w = round(imp * 380)
            fi_html += f"""
            <div class="factor-row">
              <div class="factor-name">{label}</div>
              <div class="factor-bar-wrap">
                <div class="factor-bar" style="width:{bar_w}px;"></div>
              </div>
              <div class="factor-pct">{round(imp*100,1)}%</div>
            </div>"""
        st.markdown(fi_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Customer Summary</div>',
                    unsafe_allow_html=True)
        fields = [
            ("Country", country), ("Gender", gender),
            ("Age", age),         ("Credit score", credit_score),
            ("Tenure", f"{tenure} yrs"), ("Balance", f"${balance:,}"),
            ("Products", products), ("Est. salary", f"${salary:,}"),
            ("Credit card", credit_card), ("Active member", active),
        ]
        sc1, sc2 = st.columns(2)
        for i, (k, v) in enumerate(fields):
            with (sc1 if i % 2 == 0 else sc2):
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;
                     padding:.4rem 0;border-bottom:1px solid #f1f5f9;">
                  <span style="font-size:12px;color:#94a3b8">{k}</span>
                  <span style="font-size:13px;font-weight:500;color:#0f172a">{v}</span>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analytics
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    auc        = roc_auc_score(y_test, proba_test)
    ap         = average_precision_score(y_test, proba_test)
    acc        = (preds_test == y_test.values).mean()
    churn_rate = df['churn'].mean()
    conf_scores = np.abs(proba_test - 0.5) * 2

    k1, k2, k3, k4 = st.columns(4)
    for col, label, value, sub in [
        (k1, "Test AUC",       f"{auc:.4f}",    "ROC area under curve"),
        (k2, "Avg Precision",  f"{ap:.4f}",     "Precision-recall AUC"),
        (k3, "Accuracy",       f"{acc:.1%}",    "At threshold 0.5"),
        (k4, "Churn Rate",     f"{churn_rate:.1%}", "Dataset base rate"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    def make_fig(figsize=(5.5, 4)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#f8fafc'); fig.patch.set_facecolor('white')
        ax.spines[['top','right']].set_visible(False)
        return fig, ax

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, proba_test)
        fig, ax = make_fig()
        ax.plot(fpr, tpr, color='#3b82f6', lw=2, label=f'AUC = {auc:.3f}')
        ax.fill_between(fpr, tpr, alpha=0.08, color='#3b82f6')
        ax.plot([0,1],[0,1],'--',color='#cbd5e1',lw=1)
        ax.set_xlabel('False positive rate', fontsize=11)
        ax.set_ylabel('True positive rate',  fontsize=11)
        ax.legend(fontsize=10)
        st.pyplot(fig, width='stretch'); plt.close()

    with c2:
        st.markdown('<div class="section-header">Precision-Recall Curve</div>', unsafe_allow_html=True)
        prec, rec, _ = precision_recall_curve(y_test, proba_test)
        fig, ax = make_fig()
        ax.plot(rec, prec, color='#8b5cf6', lw=2, label=f'AP = {ap:.3f}')
        ax.fill_between(rec, prec, alpha=0.08, color='#8b5cf6')
        ax.axhline(churn_rate, color='#cbd5e1', ls='--', lw=1, label=f'Baseline ({churn_rate:.2f})')
        ax.set_xlabel('Recall',    fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.legend(fontsize=10)
        st.pyplot(fig, width='stretch'); plt.close()

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, preds_test)
        fig, ax = make_fig((4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Stay','Churn'], yticklabels=['Stay','Churn'],
                    linewidths=.5, annot_kws={'size':13})
        ax.set_xlabel('Predicted', fontsize=11); ax.set_ylabel('Actual', fontsize=11)
        fig.patch.set_facecolor('white')
        st.pyplot(fig, width='stretch'); plt.close()

    with c4:
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        fi = pd.Series(model.feature_importances_, index=FEATURE_LABELS).sort_values()
        fig, ax = make_fig((5, 4.2))
        colors = ['#ef4444' if v >= fi.nlargest(2).min() else '#3b82f6' for v in fi.values]
        ax.barh(fi.index, fi.values * 100, color=colors, height=0.65)
        ax.set_xlabel('Importance (%)', fontsize=11)
        for i, v in enumerate(fi.values):
            ax.text(v*100+.3, i, f'{v*100:.1f}%', va='center', fontsize=9, color='#475569')
        st.pyplot(fig, width='stretch'); plt.close()

    c5, c6 = st.columns(2, gap="large")
    with c5:
        st.markdown('<div class="section-header">Confidence Score Distribution</div>', unsafe_allow_html=True)
        fig, ax = make_fig()
        ax.hist(proba_test[y_test == 0], bins=35, alpha=0.6, color='#22c55e', label='Actual: Stay', density=True)
        ax.hist(proba_test[y_test == 1], bins=35, alpha=0.6, color='#ef4444', label='Actual: Churn', density=True)
        ax.axvline(0.5, color='#0f172a', ls='--', lw=1.5, label='Threshold 0.5')
        ax.set_xlabel('Predicted churn probability', fontsize=11)
        ax.set_ylabel('Density', fontsize=11); ax.legend(fontsize=10)
        st.pyplot(fig, width='stretch'); plt.close()

    with c6:
        st.markdown('<div class="section-header">Churn Rate by Country & Gender</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.8))
        ct = df.groupby('country')['churn'].mean().sort_values(ascending=False)
        axes[0].bar(ct.index, ct.values*100, color=COLOR_SEQ[:3], width=0.5)
        axes[0].set_ylabel('Churn rate (%)', fontsize=10)
        axes[0].set_title('By country', fontsize=11, fontweight='500')
        for i, v in enumerate(ct.values):
            axes[0].text(i, v*100+.3, f'{v*100:.1f}%', ha='center', fontsize=9)
        cg = df.groupby('gender')['churn'].mean()
        axes[1].bar(cg.index, cg.values*100, color=['#ec4899','#3b82f6'], width=0.4)
        axes[1].set_ylabel('Churn rate (%)', fontsize=10)
        axes[1].set_title('By gender', fontsize=11, fontweight='500')
        for i, v in enumerate(cg.values):
            axes[1].text(i, v*100+.3, f'{v*100:.1f}%', ha='center', fontsize=9)
        for ax in axes:
            ax.set_facecolor('#f8fafc')
            ax.spines[['top','right']].set_visible(False)
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig, width='stretch'); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Accuracy vs Confidence Tier</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.columns(3)
    for col, label, lo, hi, color in [
        (t1, "High Confidence",   0.7, 1.01, "#22c55e"),
        (t2, "Medium Confidence", 0.4, 0.70, "#f59e0b"),
        (t3, "Low Confidence",    0.0, 0.40, "#ef4444"),
    ]:
        mask  = (conf_scores >= lo) & (conf_scores < hi)
        n     = int(mask.sum())
        acc_t = (preds_test[mask] == y_test.values[mask]).mean() if n > 0 else 0
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top:3px solid {color};">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{acc_t:.1%}</div>
              <div class="metric-sub">{n:,} predictions</div>
              <div class="conf-track" style="margin-top:.5rem;">
                <div class="conf-fill" style="width:{acc_t*100:.0f}%;background:{color};"></div>
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Full Dataset — 10,000 Customers</div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        filter_churn   = st.selectbox("Churn status", ["All", "Churned", "Stayed"])
    with fc2:
        filter_country = st.selectbox("Country", ["All"] + sorted(df['country'].unique()))
    with fc3:
        age_range = st.slider("Age range", 18, 92, (18, 92))

    dff = df.copy()
    if filter_churn   == "Churned": dff = dff[dff['churn'] == 1]
    if filter_churn   == "Stayed":  dff = dff[dff['churn'] == 0]
    if filter_country != "All":     dff = dff[dff['country'] == filter_country]
    dff = dff[(dff['age'] >= age_range[0]) & (dff['age'] <= age_range[1])]

    st.markdown(f"""
    <div style="font-size:13px;color:#64748b;margin-bottom:.75rem">
      Showing <b style="color:#0f172a">{len(dff):,}</b> of <b style="color:#0f172a">{len(df):,}</b> customers
    </div>""", unsafe_allow_html=True)

    disp = dff.reset_index(drop=True).copy()
    disp['churn']            = disp['churn'].map({0:'✅ Stay', 1:'🔴 Churn'})
    disp['balance']          = disp['balance'].apply(lambda x: f"${x:,.0f}")
    disp['estimated_salary'] = disp['estimated_salary'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(disp, width='stretch', height=420)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Summary Statistics (filtered)</div>', unsafe_allow_html=True)
    desc = dff[['credit_score','age','tenure','balance','estimated_salary']].describe().round(1).T[['mean','std','min','50%','max']]
    desc.columns = ['Mean','Std','Min','Median','Max']
    st.dataframe(desc, width='stretch')