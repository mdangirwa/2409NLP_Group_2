# app.py
import streamlit as st
import pandas as pd
import pickle
import re
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load trained components
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Download necessary NLTK assets (only once)
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function (must match training)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return " ".join(tokens)

def predict_category(text):
    cleaned = clean_text(text)
    X_input = vectorizer.transform([cleaned])
    encoded_pred = model.predict(X_input)[0]
    decoded_pred = label_encoder.inverse_transform([encoded_pred])[0]
    probabilities = model.predict_proba(X_input)[0]
    return decoded_pred, probabilities

def generate_wordcloud(text):
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# Streamlit UI Setup
st.set_page_config(page_title="News Category Classifier", layout="wide")
st.title("📰 News Article Category Classifier")
st.write("Classify news articles into categories using your trained Logistic Regression model.")

option = st.sidebar.radio("Choose input type:", ("Single Article", "Batch Prediction"))

if option == "Single Article":
    user_input = st.text_area("✍️ Enter Article Text", height=200)
    if st.button("🔍 Predict Category"):
        if user_input.strip():
            category, probabilities = predict_category(user_input)
            st.success(f"Predicted Category: **{category}**")

            st.subheader("📊 Confidence Scores")
            prob_df = pd.DataFrame({
                "Category": label_encoder.classes_,
                "Probability": probabilities
            }).sort_values(by="Probability", ascending=False)
            st.dataframe(prob_df.reset_index(drop=True))

            st.subheader("☁️ Word Cloud of Input Text")
            generate_wordcloud(user_input)
        else:
            st.warning("Please enter some text to classify.")

elif option == "Batch Prediction":
    uploaded_file = st.file_uploader("📁 Upload CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        text_columns = [col for col in df.columns if df[col].dtype == "object"]
        selected_col = st.selectbox("Select the column containing article text:", text_columns)

        if st.button("🔍 Predict Categories"):
            texts = df[selected_col].fillna("")
            cleaned_texts = texts.apply(clean_text)
            X_input = vectorizer.transform(cleaned_texts)
            encoded_preds = model.predict(X_input)
            decoded_preds = label_encoder.inverse_transform(encoded_preds)
            prob_matrix = model.predict_proba(X_input)

            df["Predicted Category"] = decoded_preds
            st.subheader("📄 Prediction Results")
            st.dataframe(df[[selected_col, "Predicted Category"]])

            st.subheader("📊 Average Confidence Scores")
            avg_probs = pd.DataFrame(prob_matrix, columns=label_encoder.classes_).mean().sort_values(ascending=False)
            st.bar_chart(avg_probs)

            st.subheader("☁️ Word Cloud of All Articles")
            generate_wordcloud(" ".join(texts))
