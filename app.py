import streamlit as st
import pandas as pd
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import altair as alt
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon (only once)
nltk.download("vader_lexicon")

# Get API key from secrets
NEWS_API_KEY ="e35c3d77cbcb402eaadcafbf5bcedac4"

# ------------------------
# Fetch news
# ------------------------
def fetch_news(topic, page_size=5):
    url = f"https://newsapi.org/v2/everything?q={topic}&pageSize={page_size}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    r = requests.get(url)
    articles = r.json().get("articles", [])
    return pd.DataFrame([{"title": a["title"]} for a in articles if a.get("title")])

# ------------------------
# Sentiment labeling (Bootstrap)
# ------------------------
def label_sentiment(df_pd):
    sia = SentimentIntensityAnalyzer()
    def classify_title(title):
        score = sia.polarity_scores(title)["compound"]
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"
    df_pd["sentiment"] = df_pd["title"].apply(classify_title)
    return df_pd

# ------------------------
# Streamlit UI
# ------------------------
st.title("📰 News Sentiment Dashboard (Scikit-learn Version)")

topic = st.text_input("Topic", "technology")
mode = st.radio("Mode", ["Bootstrap", "Predict"])
num_articles = st.slider("Number of articles", 3, 20, 5)

if st.button("Run"):
    df_pd = fetch_news(topic, num_articles)
    if df_pd.empty:
        st.error("No news found.")
    else:
        if mode == "Bootstrap":
            # Label data with VADER
            df_pd = label_sentiment(df_pd)

            # Train model
            vectorizer = TfidfVectorizer(stop_words="english")
            X = vectorizer.fit_transform(df_pd["title"])
            y = df_pd["sentiment"]

            model = LogisticRegression(max_iter=200)
            model.fit(X, y)

            # Training accuracy
            y_pred = model.predict(X)
            acc = accuracy_score(y, y_pred)

            # Save to session
            st.session_state.update({
                "model": model,
                "vectorizer": vectorizer
            })

            st.success("Model trained successfully!")
            st.info(f"Training Accuracy: {acc:.2%}")
            st.subheader("Bootstrap Data")
            st.write(df_pd[["title", "sentiment"]])

        elif mode == "Predict":
            if "model" not in st.session_state:
                st.error("Run Bootstrap first!")
            else:
                model = st.session_state["model"]
                vectorizer = st.session_state["vectorizer"]

                # Transform and predict
                X_new = vectorizer.transform(df_pd["title"])
                predictions = model.predict(X_new)

                predictions_df = df_pd.copy()
                predictions_df["sentiment"] = predictions

                st.subheader("Predictions")
                st.write(predictions_df[["title", "sentiment"]])

                # Streaming simulation
                st.subheader("Streaming Simulation")
                for idx, row in predictions_df.iterrows():
                    st.write(f"**{row['title']}** → {row['sentiment']}")
                    time.sleep(0.5)

                # Altair chart
                chart_data = predictions_df.groupby("sentiment").size().reset_index(name="count")
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x="sentiment",
                    y="count",
                    color="sentiment"
                )
                st.altair_chart(chart, use_container_width=True)
