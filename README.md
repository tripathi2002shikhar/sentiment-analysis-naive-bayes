# Sentiment Analysis of Product Reviews using Naive Bayes

## 📌 Project Overview

This project uses Natural Language Processing (NLP) and Machine Learning to analyze customer product reviews and classify them as **positive** or **negative**.

The goal is to build a simple yet powerful sentiment classifier using **TF-IDF Vectorization** and the **Naive Bayes algorithm**, with a comparison to **Logistic Regression**.

---

## 📂 Dataset

- Simulated Amazon-style reviews (custom dataset)
- Two columns: `Text` (review), `Score` (rating from 1 to 5)
- Sentiment labeling:
  - Score 4–5 → **Positive**
  - Score 1–2 → **Negative**
  - Score 3 → Ignored

File: `final_sample_reviews.csv`

---

## 🛠️ Tools & Libraries Used

- Python
- Pandas, NumPy
- Scikit-learn
- NLTK
- Matplotlib, Seaborn
- WordCloud

---

## 🔄 Workflow

1. Data Cleaning and Preprocessing
   - Lowercasing, punctuation removal
   - Stopword removal
   - Tokenization
2. TF-IDF Feature Extraction
3. Train-Test Split (70/30)
4. Model Training:
   - Multinomial Naive Bayes
   - Logistic Regression (comparison)
5. Evaluation:
   - Accuracy
   - Precision, Recall, F1-score
   - Confusion Matrix
6. Sample Predictions

---

## 📊 EDA Highlights

- Sentiment Distribution Pie Chart
- WordClouds of Positive & Negative Reviews

---

## 🧪 Model Performance (Example)

| Model                | Accuracy |
|---------------------|----------|
| Naive Bayes          | 86%      |
| Logistic Regression  | 88%      |

---

## 📝 Sample Predictions

```text
"I love this product!" → Positive  
"Worst purchase ever." → Negative  
"Works okay, nothing special." → (not classified - neutral ignored)
