import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from openai import OpenAI
import os

# Page Config
st.set_page_config(
    page_title="IDS - XAI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:       #0d0d0d;
    --surface:  #141414;
    --surface2: #1c1c1c;
    --surface3: #242424;
    --amber:    #f59e0b;
    --red:      #ef4444;
    --green:    #22c55e;
    --muted:    #6b7280;
    --text:     #f3f4f6;
    --border:   rgba(245,158,11,0.18);
    --border2:  rgba(255,255,255,0.06);
}

html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'Poppins', sans-serif !important;
    color: var(--text) !important;
}
.stApp > header { display: none; }

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.stNumberInput input {
    background: var(--surface3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stNumberInput input:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px rgba(245,158,11,0.15) !important;
}
.stNumberInput label, .stRadio label, .stMultiSelect label {
    font-size: 0.72rem !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
}

/* Multiselect */
.stMultiSelect > div > div {
    background: var(--surface3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
}
.stMultiSelect span[data-baseweb="tag"] {
    background: rgba(245,158,11,0.15) !important;
    color: var(--amber) !important;
    border: 1px solid rgba(245,158,11,0.3) !important;
    border-radius: 6px !important;
    font-size: 0.72rem !important;
}

/* Primary button */
.stButton > button {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
    color: #0d0d0d !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(245,158,11,0.35) !important;
}
button[data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    color: var(--amber) !important;
    border: 1px solid rgba(245,158,11,0.4) !important;
    border-radius: 10px !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
button[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(245,158,11,0.08) !important;
}

.stSpinner > div { border-top-color: var(--amber) !important; }

.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border: 1px solid rgba(245,158,11,0.2) !important;
    border-radius: 10px !important;
    color: var(--amber) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent {
    background: var(--surface2) !important;
    border: 1px solid rgba(245,158,11,0.1) !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    padding: 1rem !important;
}

div[data-testid="stMarkdownContainer"] p { color: var(--text); font-size: 0.88rem; }
</style>
""", unsafe_allow_html=True)

# Load Assets
@st.cache_resource
def load_all():
    model = joblib.load("models/LightGBM.pkl")
    le    = joblib.load("models/label_encoder.pkl")
    try:
        explainer = joblib.load("models/shap_explainer.pkl")
    except FileNotFoundError:
        explainer = shap.TreeExplainer(model)
    with open("data/feature_columns.json") as f:
        features = json.load(f)
    return model, le, explainer, features

model, le, explainer, features = load_all()

# Helpers
def severity(label):
    high   = {"DoS Hulk","DDoS","DoS GoldenEye","DoS slowloris","DoS Slowhttptest","Heartbleed"}
    medium = {"Bot","PortScan","FTP-Patator","SSH-Patator","Web Attack \x96 Brute Force",
              "Web Attack \x96 Sql Injection","Web Attack \x96 XSS","Infiltration"}
    if label in high:     return "HIGH",   "#ef4444"
    if label in medium:   return "MEDIUM", "#f59e0b"
    if label == "BENIGN": return "BENIGN", "#22c55e"
    return "LOW", "#a78bfa"

def openai_explain(prediction, top_features, feature_values, confidence):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None, "⚠️ OPENAI_API_KEY belum diset di environment / repo secret."
    feat_lines = "\n".join(f"  - {f}" for f in top_features)
    val_lines  = "\n".join(
        f"  - {k}: {v:.4g}" for k, v in list(feature_values.items())[:15] if v != 0
    )
    prompt = f"""Kamu adalah sistem penjelasan otomatis untuk Intrusion Detection System berbasis machine learning.

Hasil deteksi model LightGBM pada traffic jaringan:
- Prediksi   : {prediction}
- Confidence : {confidence:.2%}

Top 5 fitur SHAP yang paling berpengaruh:
{feat_lines}

Nilai fitur traffic (non-zero):
{val_lines}

Tulis penjelasan 3–4 kalimat dalam bahasa Indonesia yang profesional dan naratif (tanpa bullet points):
1. Mengapa model memutuskan "{prediction}"
2. Apa yang mencurigakan atau normal dari fitur-fitur tersebut — jelaskan maknanya secara konkret \
(misal: "Flow IAT Mean rendah berarti paket dikirim sangat cepat dan berdekatan")
3. Rekomendasi tindakan singkat yang perlu dilakukan jika ini adalah serangan
Maksimal 150 kata."""
    try:
        resp = OpenAI(api_key=api_key).chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Kamu adalah analis keamanan jaringan senior."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=300, temperature=0.4,
        )
        return resp.choices[0].message.content, None
    except Exception as e:
        return None, f"OpenAI error: {e}"

def plot_shap_bar(shap_2d, feature_names, pred_class, top_n=10):
    vals  = shap_2d[:, pred_class]
    pairs = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names, values = [p[0] for p in pairs], [p[1] for p in pairs]
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in values]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor("#141414")
    ax.set_facecolor("#141414")
    ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.55)
    ax.axvline(0, color="#2a2a2a", linewidth=1.2)
    ax.set_xlabel("SHAP value", color="#6b7280", fontsize=8)
    ax.tick_params(colors="#9ca3af", labelsize=7.5)
    for sp in ax.spines.values(): sp.set_edgecolor("#1f1f1f")
    ax.grid(axis="x", color="#1f1f1f", linewidth=0.6)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color="#ef4444", label="↑ mengarah ke attack"),
                 Patch(color="#22c55e", label="↓ mengarah ke benign")],
        fontsize=7, facecolor="#1c1c1c", edgecolor="#2a2a2a",
        labelcolor="#9ca3af", loc="lower right"
    )
    plt.tight_layout(pad=0.5)
    return fig

def section_label(text, color="#f59e0b"):
    st.markdown(
        f'<div style="font-size:0.68rem;color:{color};letter-spacing:0.12em;'
        f'font-weight:600;margin-bottom:0.6rem">{text}</div>',
        unsafe_allow_html=True
    )

def sub_label(text):
    st.markdown(
        f'<div style="font-size:0.62rem;color:#6b7280;letter-spacing:0.08em;'
        f'margin:0.6rem 0 0.4rem">{text}</div>',
        unsafe_allow_html=True
    )

# Top SHAP features
TOP_SHAP_FEATURES = [
    "Flow IAT Mean",
    "Flow IAT Min",
    "Bwd Packets/s",
    "Fwd Header Length",
    "Init_Win_bytes_forward",
]
CONTEXT_FEATURES = [
    "Active Max",
    "Init_Win_bytes_bckward",
    "Destination Port",
    "Fwd IAT Std",
    "Flow Duration",
    "min_seg_size_forward",
]
DISPLAY_FEATURES = TOP_SHAP_FEATURES + CONTEXT_FEATURES

# PRESETS
@st.cache_data
def load_presets():
    try:
        with open("data/preset_samples.json") as f:
            raw = json.load(f)
        return {
            "Custom":         {},
            "Normal Traffic": raw.get("BENIGN", {}),
            "DoS Hulk":       raw.get("DoS Hulk", {}),
            "PortScan":       raw.get("PortScan", {}),
            "Bot":            raw.get("Bot", {}),
        }
    except FileNotFoundError:
        return {"Custom": {}}

PRESETS      = load_presets()
PRESET_NAMES = list(PRESETS.keys())

def sk(feat):
    """Safe session_state key for a feature."""
    return "inp_" + feat.replace(" ", "_").replace("/", "_per_").replace("-", "_").replace(".", "_")

# SIDEBAR
with st.sidebar:
    st.markdown("""
<div style="padding:0.5rem 0 1.4rem">
  <div style="font-family:'Poppins',sans-serif;font-size:1.3rem;font-weight:800;
              background:linear-gradient(90deg,#f59e0b,#fde68a);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent">
    🛡️ IDS · XAI
  </div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.63rem;
              color:#6b7280;letter-spacing:0.12em;margin-top:0.25rem">
    INTRUSION DETECTION SYSTEM
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:0.68rem;color:#6b7280;letter-spacing:0.1em;'
        'font-weight:600;margin-bottom:0.5rem">⚡ QUICK PRESET</div>',
        unsafe_allow_html=True
    )
    preset = st.radio("", PRESET_NAMES, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div style="background:#1c1c1c;border:1px solid rgba(245,158,11,0.12);border-radius:10px;padding:1rem">
  <div style="font-size:0.63rem;color:#6b7280;letter-spacing:0.1em;font-weight:600;margin-bottom:0.6rem">MODEL INFO</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#9ca3af;line-height:2.1">
    Algorithm &nbsp;: LightGBM<br>
    Dataset &nbsp;&nbsp;&nbsp;: CIC-IDS-2017<br>
    Balancing &nbsp;: SMOTE<br>
    Classes &nbsp;&nbsp;&nbsp;: 15<br>
    Accuracy &nbsp;&nbsp;: 99.8%<br>
    Macro F1 &nbsp;&nbsp;: 0.905<br>
    Macro Prec &nbsp;: 0.878<br>
    Macro Rec &nbsp;&nbsp;: 0.953<br>
    XAI &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: SHAP<br>
    LLM &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: GPT-4o-mini
  </div>
</div>
""", unsafe_allow_html=True)

# Inject preset
if "last_preset" not in st.session_state:
    st.session_state["last_preset"] = preset
    st.session_state["preset_full"] = PRESETS.get(preset, {})

if st.session_state["last_preset"] != preset:
    p = PRESETS.get(preset, {})
    for feat in DISPLAY_FEATURES:
        st.session_state[sk(feat)] = float(p.get(feat, 0.0))
    st.session_state["preset_full"] = p
    st.session_state["last_preset"] = preset

# HEADER
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem">
  <div>
    <h1 style="font-family:'Poppins',sans-serif;font-weight:800;font-size:2rem;margin:0;
               background:linear-gradient(90deg,#f59e0b,#fde68a,#f59e0b);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent">
      Intrusion Detection System
    </h1>
    <p style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#6b7280;
              letter-spacing:0.1em;margin:0.3rem 0 0">
      EXPLAINABLE AI DASHBOARD · CIC-IDS-2017 · LIGHTGBM + SHAP + GPT-4o-mini
    </p>
  </div>
  <div style="display:flex;gap:0.4rem;flex-wrap:wrap;justify-content:flex-end">
    <span style="background:rgba(245,158,11,0.12);color:#f59e0b;border:1px solid rgba(245,158,11,0.25);border-radius:999px;padding:0.2rem 0.75rem;font-size:0.68rem;font-weight:600">LightGBM</span>
    <span style="background:rgba(34,197,94,0.1);color:#22c55e;border:1px solid rgba(34,197,94,0.25);border-radius:999px;padding:0.2rem 0.75rem;font-size:0.68rem;font-weight:600">SHAP</span>
    <span style="background:rgba(167,139,250,0.1);color:#a78bfa;border:1px solid rgba(167,139,250,0.25);border-radius:999px;padding:0.2rem 0.75rem;font-size:0.68rem;font-weight:600">GPT-4o-mini</span>
  </div>
</div>
<hr style="border:none;border-top:1px solid rgba(245,158,11,0.12);margin-bottom:1.5rem">
""", unsafe_allow_html=True)

# BASIC INPUT
section_label("📡 NETWORK TRAFFIC FEATURES")
sub_label("TOP SHAP FEATURES")

shap_cols = st.columns(len(TOP_SHAP_FEATURES))
display_vals = {}
for i, feat in enumerate(TOP_SHAP_FEATURES):
    with shap_cols[i]:
        display_vals[feat] = st.number_input(
            feat, step=0.01, format="%.2f", key=sk(feat)
        )

sub_label("TRAFFIC CONTEXT")
ctx_cols = st.columns(len(CONTEXT_FEATURES))
for i, feat in enumerate(CONTEXT_FEATURES):
    with ctx_cols[i]:
        display_vals[feat] = st.number_input(
            feat, step=0.01, format="%.2f", key=sk(feat)
        )

# ADVANCED INPUT
extra_features = [f for f in features if f not in DISPLAY_FEATURES]

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("⚙️  Advanced Input", expanded=False):
    st.markdown(
        '<div style="font-size:0.72rem;color:#6b7280;margin-bottom:1rem;line-height:1.6">'
        'Pilih fitur tambahan yang ingin kamu atur nilainya secara manual. '
        'Jika preset aktif, nilai fitur lain sudah terisi otomatis dari dataset.'
        '</div>',
        unsafe_allow_html=True
    )
    selected_extra = st.multiselect(
        "Pilih fitur tambahan:",
        options=extra_features,
        default=[],
        placeholder="Cari nama fitur...",
        key="adv_selected"
    )
    adv_override = {}
    if selected_extra:
        st.markdown("<br>", unsafe_allow_html=True)
        adv_cols = st.columns(3)
        for i, feat in enumerate(selected_extra):
            safe_key = (
                "adv_"
                + feat.replace(" ", "_").replace("/", "_per_")
                      .replace("-", "_").replace(".", "_")
            )
            preset_val = float(st.session_state.get("preset_full", {}).get(feat, 0.0))
            with adv_cols[i % 3]:
                adv_override[feat] = st.number_input(
                    feat, value=preset_val, step=0.01, format="%.4f", key=safe_key
                )
    else:
        st.markdown(
            '<div style="text-align:center;color:#374151;font-size:0.8rem;padding:1rem 0">'
            'Belum ada fitur dipilih</div>',
            unsafe_allow_html=True
        )

# BUTTONS
st.markdown("<br>", unsafe_allow_html=True)
run = st.button("🔍  Analyze Traffic", use_container_width=True)

# BUILD FEATURE VECTOR
def build_row():
    preset_full = st.session_state.get("preset_full", {})
    full = {f: float(preset_full.get(f, 0.0)) for f in features}
    full.update(display_vals)
    full.update(adv_override)
    row     = pd.DataFrame([full])[features]
    display = {k: v for k, v in full.items() if v != 0.0}
    return row, display

# RENDER RESULT
def render_result(row, display_map, mode_label=""):
    pred_enc = model.predict(row)[0]
    proba    = model.predict_proba(row)[0]
    conf     = proba[pred_enc]
    label    = le.inverse_transform([pred_enc])[0]
    sev, sev_color = severity(label)
    is_attack = label != "BENIGN"

    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(245,158,11,0.12);margin:1.5rem 0'>",
        unsafe_allow_html=True
    )

    badge = (
        f'<span style="background:rgba(167,139,250,0.12);color:#a78bfa;'
        f'border:1px solid rgba(167,139,250,0.3);border-radius:999px;'
        f'padding:0.15rem 0.6rem;font-size:0.63rem;font-weight:600;margin-left:0.5rem">'
        f'{mode_label}</span>'
    ) if mode_label else ""
    section_label(f"🔬 ANALYSIS RESULT {badge}")

    col_pred, col_shap = st.columns([1, 2], gap="large")

    # Left column: classification card + probabilities
    with col_pred:
        icon = "🚨" if is_attack else "✅"
        bg   = "rgba(239,68,68,0.07)"  if is_attack else "rgba(34,197,94,0.07)"
        bord = "#ef4444"               if is_attack else "#22c55e"
        st.markdown(f"""
<div style="background:{bg};border:1px solid {bord}22;border-left:3px solid {bord};
            border-radius:12px;padding:1.4rem;text-align:center;margin-bottom:1rem">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.63rem;color:#6b7280;letter-spacing:0.12em">CLASSIFICATION</div>
  <div style="font-family:'Poppins',sans-serif;font-size:1.55rem;font-weight:800;color:{bord};margin:0.5rem 0">{icon} {label}</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.77rem;color:#9ca3af">confidence: {conf:.2%}</div>
  <div style="margin-top:0.9rem">
    <span style="background:{sev_color}18;color:{sev_color};border:1px solid {sev_color}33;
                 border-radius:999px;padding:0.22rem 0.9rem;font-size:0.68rem;font-weight:700;letter-spacing:0.08em">
      ⚡ {sev}
    </span>
  </div>
</div>""", unsafe_allow_html=True)

        sub_label("CLASS PROBABILITIES")
        prob_df = (
            pd.DataFrame({"Class": le.classes_, "Prob": proba})
            .sort_values("Prob", ascending=False)
            .head(5)
        )
        fig2, ax2 = plt.subplots(figsize=(4, 2.5))
        fig2.patch.set_facecolor("#141414")
        ax2.set_facecolor("#141414")
        bar_c = ["#f59e0b" if c == label else "#242424" for c in prob_df["Class"]]
        ax2.barh(prob_df["Class"][::-1], prob_df["Prob"][::-1], color=bar_c[::-1], height=0.5)
        ax2.tick_params(colors="#9ca3af", labelsize=7)
        for sp in ax2.spines.values(): sp.set_edgecolor("#1f1f1f")
        ax2.grid(axis="x", color="#1f1f1f", linewidth=0.5)
        ax2.set_xlim(0, 1)
        fig2.tight_layout(pad=0.4)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # Right column: SHAP bar + top features
    with col_shap:
        sub_label("SHAP FEATURE IMPORTANCE")
        with st.spinner("Computing SHAP values…"):
            sv = explainer.shap_values(row)
            if isinstance(sv, np.ndarray) and sv.ndim == 3:
                shap_2d = sv[0]
            elif isinstance(sv, list):
                shap_2d = np.array([s[0] for s in sv]).T
            else:
                shap_2d = sv

        fig_shap = plot_shap_bar(shap_2d, features, pred_enc)
        st.pyplot(fig_shap, use_container_width=True)
        plt.close(fig_shap)

        vals  = shap_2d[:, pred_enc]
        pairs = sorted(zip(features, vals), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_features_text = [
            f"{f} ({'increase' if v > 0 else 'decrease'}, impact={v:.3f})"
            for f, v in pairs
        ]
        sub_label("TOP CONTRIBUTING FEATURES")
        for tf in top_features_text:
            dot = "🔴" if "increase" in tf else "🟢"
            st.markdown(
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.74rem;'
                f'color:#9ca3af;padding:0.3rem 0;border-bottom:1px solid #1f1f1f">{dot} {tf}</div>',
                unsafe_allow_html=True
            )

    # LLM explanation
    st.markdown("<br>", unsafe_allow_html=True)
    section_label("🤖 LLM NATURAL LANGUAGE EXPLANATION")
    with st.spinner("Generating explanation via GPT-4o-mini…"):
        explanation, err = openai_explain(label, top_features_text, display_map, conf)
    if err:
        st.warning(err)
    else:
        st.markdown(f"""
<div style="background:#1c1c1c;border:1px solid rgba(245,158,11,0.15);border-left:3px solid #f59e0b;
            border-radius:10px;padding:1.25rem 1.5rem;font-size:0.88rem;line-height:1.8;color:#d1d5db">
{explanation}
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TRIGGER
# ═══════════════════════════════════════════════════════════════════════════════
if run:
    row, dmap = build_row()
    is_adv = bool(adv_override)
    render_result(row, dmap, mode_label="ADVANCED" if is_adv else "")