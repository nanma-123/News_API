# app.py
import streamlit as st
import pandas as pd
import requests
from transformers import pipeline

# --- App title ---
st.title("📰 News Sentiment Analyzer")

# --- Input field ---
topic = st.text_input("Enter a news topic:", "Artificial Intelligence")

if st.button("Analyze Sentiment"):
    st.write(f"Fetching news for: **{topic}** ...")

    # You can use GNews API or NewsAPI — here’s an example using GNews
    url = f"https://gnews.io/api/v4/search?q={topic}&lang=en&country=in&max=5&apikey=YOUR_API_KEY"
    response = requests.get(url)
    data = response.json()

    if "articles" in data:
        articles = pd.DataFrame(data["articles"])[["title", "description", "url"]]

        # Load sentiment model
        sentiment_pipeline = pipeline("sentiment-analysis")

        st.subheader("Results")
        sentiments = []
        for i, row in articles.iterrows():
            text = row["description"] or row["title"]
            result = sentiment_pipeline(text[:512])[0]
            sentiments.append(result["label"])

        articles["sentiment"] = sentiments
        st.dataframe(articles)

        st.bar_chart(articles["sentiment"].value_counts())
    else:
        st.error("No news found or invalid API key.")
