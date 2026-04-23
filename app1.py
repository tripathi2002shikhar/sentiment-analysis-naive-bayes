# app.py — Enhanced Sentiment Analysis UI
import streamlit as st
import pickle
import re
import string
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image

nltk.download("stopwords", quiet=True)

# ======== CONFIG ========
DATASET_PATH = "reviews.csv"   
MODEL_FOLDER = "models1"                              
LR_MODEL_FILE = f"{MODEL_FOLDER}/sentiment_lr_model.pkl"
NB_MODEL_FILE = f"{MODEL_FOLDER}/sentiment_nb_model.pkl"
VECT_FILE = f"{MODEL_FOLDER}/tfidf_vectorizer.pkl"

# ======== UTILITIES ========
@st.cache_resource
def load_models():
    with open(LR_MODEL_FILE, "rb") as f:
        lr = pickle.load(f)
    with open(NB_MODEL_FILE, "rb") as f:
        nb = pickle.load(f)
    with open(VECT_FILE, "rb") as f:
        tfidf = pickle.load(f)
    return lr, nb, tfidf

def clean_text(text: str) -> str:
    # normalize quotes
    text = str(text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.lower()
    # preserve n't -> not
    text = re.sub(r"n\'t", " not", text)
    # remove apostrophes
    text = re.sub(r"'", "", text)
    # remove digits & extra whitespace & non-word characters but keep emojis as they are
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF]", " ", text)
    text = " ".join(text.split())
    return text

def safe_predict_proba(model, vec):
    """Return (sentiment, confidence) using safe index for 'positive'"""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        classes = list(model.classes_)
        if "positive" in classes:
            pos_idx = classes.index("positive")
        elif "pos" in classes:
            pos_idx = classes.index("pos")
        else:
            pos_idx = 1 if len(classes) > 1 else 0
        confidence = float(probs[pos_idx])
        sentiment = "positive" if confidence >= threshold_slider else "negative"
        return sentiment, confidence
    else:
        pred = model.predict(vec)[0]
        return pred, None

def get_top_tokens_lr(lr_model, tfidf, n=10):
    """Return top n positive and negative tokens from Logistic Regression."""
    try:
        vocab = tfidf.vocabulary_  # token -> index
        inv_vocab = {i: t for t, i in vocab.items()}
        coefs = lr_model.coef_[0]  # binary classification expected
        # pair (token, coef)
        pairs = [(inv_vocab[i], float(coefs[i])) for i in range(len(coefs)) if i in inv_vocab]
        pairs_sorted_pos = sorted(pairs, key=lambda x: -x[1])[:n]
        pairs_sorted_neg = sorted(pairs, key=lambda x: x[1])[:n]
        return pairs_sorted_pos, pairs_sorted_neg
    except Exception:
        return [], []

def plot_wordcloud(text):
    if not text or len(text.strip()) == 0:
        st.info("Enter text to see wordcloud.")
        return
    wc = WordCloud(width=600, height=300, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

def download_button_file(path, label="Download dataset"):
    try:
        with open(path, "rb") as f:
            data = f.read()
        st.download_button(label, data, file_name=path.split("/")[-1], mime="text/csv")
    except Exception as e:
        st.warning(f"Could not find dataset at `{path}`. Error: {e}")

# ======== LOAD ========
with st.spinner("Loading models and vectorizer..."):
    try:
        lr_model, nb_model, tfidf = load_models()
    except FileNotFoundError as e:
        st.error(f"Model files not found in `{MODEL_FOLDER}`. Make sure models are present. ({e})")
        st.stop()

# ======== LAYOUT ========
st.set_page_config(page_title="Sentiment Analyzer (Enhanced)", layout="wide", initial_sidebar_state="expanded")
st.title("🧠 Sentiment Analyzer — Interactive & Explainable")
st.markdown(
    "Paste a review, choose a model, and get an interpretable prediction. "
    "This app includes confidence, top tokens, wordcloud, and example reviews to help present a robust demo to mentors."
)

# Sidebar controls
with st.sidebar:
    st.header("Settings & Examples")
    st.markdown("**Data:**")
    st.markdown(f"- Dataset (local path): `{DATASET_PATH}`")
    download_button_file(DATASET_PATH, "⬇️ Download dataset (local path)")

    st.markdown("---")
    st.markdown("**Model selection**")
    model_choice = st.selectbox("Choose model", ["Logistic Regression", "Naive Bayes"])
    st.markdown("**Threshold**")
    threshold_slider = st.slider("Confidence threshold for positive", 0.0, 1.0, 0.55, 0.01)

    st.markdown("---")
    st.markdown("**Example reviews (one-click)**")
    examples = [
        "The product is worthless",
        "This is the best purchase I've made",
        "Disappointed with the product",
        "Worth it — excellent quality and value for money",
        "Stopped working after a week, will return"
    ]
    ex = st.selectbox("Pick example to paste into input", ["(none)"] + examples)
    if st.button("Use example"):
        st.session_state["example_text"] = ex if ex != "(none)" else ""

    st.markdown("---")
    st.markdown("**Explainability**")
    top_n = st.number_input("Top tokens to show", min_value=3, max_value=30, value=8, step=1)

# ======== MAIN UI ========
col1, col2 = st.columns([2, 1])

with col1:
    input_text = st.text_area("Enter your review here:", height=180, key="input_text")
    # if example selected in sidebar, populate textarea
    if "example_text" in st.session_state:
        if st.session_state["example_text"]:
            st.session_state["input_text"] = st.session_state["example_text"]

    c1, c2, c3 = st.columns(3)
    if c1.button("Analyze"):
        user_text = st.session_state.get("input_text", "")
        if not user_text or len(user_text.strip()) == 0:
            st.warning("Please enter a review before analyzing.")
        else:
            cleaned = clean_text(user_text)
            vec = tfidf.transform([cleaned])

            # choose model
            model = lr_model if model_choice == "Logistic Regression" else nb_model

            # compute prediction
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(vec)[0]
                classes = list(model.classes_)
                # find pos idx
                pos_idx = classes.index("positive") if "positive" in classes else (1 if len(classes) > 1 else 0)
                pos_conf = float(probs[pos_idx])
                pred_label = "positive" if pos_conf >= threshold_slider else "negative"
            else:
                pred_label = model.predict(vec)[0]
                pos_conf = None

            # UI display
            st.markdown("### 🔎 Result")
            if pos_conf is not None:
                st.metric(label="Predicted Sentiment", value=pred_label.upper(), delta=f"Confidence: {pos_conf:.2f}")
                # Progress bar for confidence
                st.progress(min(max(pos_conf, 0.0), 1.0))
            else:
                st.write(f"**Predicted:** {pred_label}")

            # Wordcloud
            st.markdown("### 🖼️ WordCloud of input")
            plot_wordcloud(cleaned)

            # Top tokens explanation (only for Logistic Regression)
            if model_choice == "Logistic Regression":
                pos_tokens, neg_tokens = get_top_tokens_lr(lr_model, tfidf, n=top_n)
                st.markdown("### 🧾 Top tokens (Logistic Regression)")
                colp, coln = st.columns(2)
                with colp:
                    st.subheader("Positive tokens")
                    for tok, coef in pos_tokens:
                        st.write(f"+ {tok} ({coef:.3f})")
                with coln:
                    st.subheader("Negative tokens")
                    for tok, coef in neg_tokens:
                        st.write(f"- {tok} ({coef:.3f})")
            else:
                st.info("Token-level coefficients are shown only for Logistic Regression.")

            # Raw probability table
            st.markdown("### 📊 Raw probabilities")
            if hasattr(model, "predict_proba"):
                prob_map = {cls: float(p) for cls, p in zip(model.classes_, probs)}
                st.json(prob_map)

    if c2.button("Clear"):
        st.session_state["input_text"] = ""

    if c3.button("Copy to clipboard (for sharing)"):
        # for demonstration we show text to copy
        txt = st.session_state.get("input_text", "")
        if txt:
            st.write("Copy the text below and share it:")
            st.code(txt)
        else:
            st.warning("No text to copy.")

with col2:
    st.markdown("## 🔎 Quick Tips")
    st.markdown("""
    - Use short reviews like *"worthless"* or *"worth it"* to test edge cases.
    - If prediction seems wrong, try lowering the threshold in the sidebar.
    - Logistic Regression shows token-level explainability.
    - You can download the dataset from the sidebar to inspect training data.
    """)
    st.markdown("---")
    st.markdown("## ℹ️ Model info")
    st.write(f"Logistic Regression classes: `{list(lr_model.classes_)}`")
    st.write(f"Naive Bayes classes: `{list(nb_model.classes_)}`")
    st.markdown("---")
    st.markdown("## 💡 Demo screenshots (for mentor slides)")
    st.image(Image.new("RGB", (640,120), color=(255,255,255)), caption="Use screenshots from the app to show UX")
    st.markdown("---")
    st.markdown("Made with ❤️ — focus on clarity & explainability for reviewers.")

# Footer
st.markdown("---")
st.caption("Note: The dataset path shown uses the session-local path. If deploying elsewhere, update DATASET_PATH and MODEL_FOLDER accordingly.")
