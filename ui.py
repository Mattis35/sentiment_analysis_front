# ui.py — Streamlit Interface for the Sentiment Analysis API

import os
import time
import requests
import streamlit as st
import plotly.express as px

# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="💬 Sentiment Analysis — UI",
    page_icon="💬",
    layout="centered",
)

# =========================
# API Configuration
# =========================
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

# =========================
# Helpers
# =========================
def validate_text(text: str) -> bool:
    return isinstance(text, str) and (1 <= len(text) <= 280)

def length_color(n: int) -> str:
    # Green (<240), Yellow (240-280), Red (>280)
    if n <= 240:
        return "#22c55e"
    if n <= 280:
        return "#f59e0b"
    return "#ef4444"

def badge(text: str, color: str) -> None:
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:4px 10px;
            border-radius:999px;
            background:{color}20;
            color:{color};
            font-weight:600;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data(ttl=15)
def get_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            return r.json(), None
        return None, f"API status: {r.status_code}"
    except Exception as e:
        return None, str(e)

def call_prediction_api(text: str):
    try:
        r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)
    except requests.exceptions.RequestException as e:
        return None, str(e)

def call_explain_api(text: str):
    try:
        r = requests.post(f"{API_URL}/explain", json={"text": text}, timeout=120)
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)
    except requests.exceptions.RequestException as e:
        return None, str(e)

def display_results(pred_data: dict):
    sentiment = pred_data.get("sentiment")
    confidence = float(pred_data.get("confidence", 0.0))
    prob_pos = float(pred_data.get("probability_positive", 0.0))
    prob_neg = float(pred_data.get("probability_negative", 0.0))

    if sentiment == "Positif":
        st.success(f"😊 **POSITIVE** ({confidence:.1%})")
    elif sentiment == "Négatif":
        st.error(f"😞 **NEGATIVE** ({confidence:.1%})")
    else:
        st.warning(f"Unexpected result: {sentiment}")

    fig = px.bar(
        x=["Negative", "Positive"],
        y=[prob_neg, prob_pos],
        labels={"x": "Class", "y": "Probability"},
        title="Predicted Probabilities",
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## ⚙️ System & API Status")
    health, err = get_health()
    if health:
        status_color = "#22c55e" if health.get("status") == "ok" else "#ef4444"
        badge(f"API: {health.get('status','unknown')}", status_color)
        st.caption(f"Model loaded: **{health.get('model_loaded')}** | Vectorizer: **{health.get('vectorizer_loaded')}**")
        if health.get("model_path"):
            st.caption(f"Model: `{health['model_path']}`")
        if health.get("vectorizer_path"):
            st.caption(f"Vectorizer: `{health['vectorizer_path']}`")
        st.caption(f"🕒 {health.get('server_time','')}")
    else:
        badge("API: unavailable", "#ef4444")
        if err:
            st.caption(f"Error: {err}")

    st.markdown("---")
    st.markdown("## 🧪 Quick Examples")
    examples = {
        "👍 Positive": "I absolutely love this product, it's fantastic!",
        "👎 Negative": "This service is awful and disappointing.",
        "😐 Neutral": "Decent quality, nothing exceptional though."
    }
    for label, txt in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["tweet_text"] = txt

    st.markdown("---")
    st.markdown("## 📝 Guide")
    st.markdown(
        "- Enter a short text (≤ 280 characters)\n"
        "- Click **Predict** to get the sentiment\n"
        "- Click **LIME** to explain the prediction (takes ~30–60s)\n"
        "- Probabilities should roughly sum to 1.00"
    )

# =========================
# Main Body
# =========================
st.markdown("<h1 style='text-align:center'>💬 Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("Enter a short piece of text (like a tweet) to analyze its **sentiment**.")

# Text input area
tweet_text = st.text_area(
    "Text",
    value=st.session_state.get("tweet_text", ""),
    max_chars=280,
    placeholder="E.g. I love this product! Fast, reliable, and top quality ✨",
    help="Maximum length: 280 characters (like Twitter).",
)

# Real-time character counter
n = len(tweet_text or "")
badge(f"{n}/280 characters", length_color(n))

text_valid = validate_text(tweet_text)

# Action buttons
cols = st.columns([1, 1, 2])
with cols[0]:
    predict_btn = st.button("🎯 Predict Sentiment", type="primary", disabled=not text_valid)
with cols[1]:
    explain_btn = st.button("🔍 LIME (30–60s)", disabled=not text_valid)

st.markdown("---")

# Prediction display
if predict_btn:
    with st.spinner("Predicting sentiment..."):
        status, data = call_prediction_api(tweet_text)

    if status == 200 and isinstance(data, dict):
        st.subheader("Result")
        display_results(data)
        st.session_state["last_prediction"] = data
        st.session_state["last_text"] = tweet_text
    elif status is None:
        st.error(f"Connection error to API: {data}")
    else:
        st.error(f"API /predict error: {status}\n\n{data}")

# LIME explanation
if explain_btn:
    with st.spinner("Running LIME explanation (this may take a while)..."):
        status, data = call_explain_api(tweet_text)

    if status == 200 and isinstance(data, dict):
        last_pred = st.session_state.get("last_prediction")
        if last_pred and st.session_state.get("last_text") == tweet_text:
            st.subheader("Result (from previous prediction)")
            display_results(last_pred)

        st.subheader("🧩 LIME Explanation")
        html = data.get("html_explanation", "")
        if isinstance(html, str) and len(html) > 0:
            st.components.v1.html(html, height=450, scrolling=True)
        else:
            st.info("No HTML explanation provided.")

        expl_list = data.get("explanation", [])
        if isinstance(expl_list, list) and len(expl_list) > 0:
            with st.expander("📊 Word Importance (list view)"):
                for item in expl_list:
                    word = item.get("word", "")
                    weight = float(item.get("weight", 0.0))
                    color = "#22c55e" if weight >= 0 else "#ef4444"
                    st.markdown(f"- <span style='color:{color}'>{word}</span>: **{weight:.4f}**", unsafe_allow_html=True)
    elif status == 503:
        st.warning("LIME is not available on the server (503).")
    elif status is None:
        st.error(f"Connection error to API: {data}")
    else:
        st.error(f"API /explain error: {status}\n\n{data}")

# Footer
st.markdown("---")
st.caption(
    f"API: `{API_URL}` · "
    "This interface uses Plotly for charts and LIME for explainability (if available)."
)
