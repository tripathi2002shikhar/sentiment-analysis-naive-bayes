import streamlit as st
import pickle
import re
import string
import nltk

nltk.download('stopwords')

# Load models
with open("models1/sentiment_lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("models1/sentiment_nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("models1/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"'", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.strip()
    return text

# Custom prediction with threshold
def predict_sentiment(model, text, threshold=0.55):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        sentiment = "positive" if probs[1] >= threshold else "negative"
        return sentiment, probs[1]
    else:
        sentiment = model.predict(vec)[0]
        return sentiment, None

# Streamlit UI
st.title("📝 Sentiment Analysis App")
st.write("Enter a product review to check if it's **Positive** or **Negative**.")

user_input = st.text_area("Enter your review here:")

model_choice = st.radio("Choose Model:", ["Logistic Regression", "Naive Bayes"])

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        if model_choice == "Logistic Regression":
            sentiment, confidence = predict_sentiment(lr_model, user_input)
        else:
            sentiment, confidence = predict_sentiment(nb_model, user_input)
        
        st.subheader("Prediction:")
        if sentiment == "positive":
            if confidence is not None:
                st.success(f"Positive Review (Confidence: {confidence:.2f})")
            else:
                st.success(f"Positive Review")
        else:
            if confidence is not None:
                st.error(f"Negative Review (Confidence: {1-confidence:.2f})")
            else:
                st.error(f"Negative Review")
    else:
        st.warning("Please enter some text before predicting.")
