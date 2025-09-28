import streamlit as st
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Real-Time News Sentiment", layout="wide")
st.title("📰 Real-Time News Sentiment Dashboard")

# Refresh interval in milliseconds (Streamlit autorefresh)
REFRESH_INTERVAL = 30 * 1000  # 30 seconds
st_autorefresh = st.experimental_autorefresh(interval=REFRESH_INTERVAL, key="refresh")

# GNews API Key
API_KEY = "API_KEY"  # <-- Replace with your key
URL = f"https://gnews.io/api/v4/top-headlines?country=in&max=10&apikey={API_KEY}"

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Sentiment function
def get_sentiment(text: str) -> str:
    score = analyzer.polarity_scores(text)["compound"]
    return "Positive" if score >= 0 else "Negative"

# Function to color-code sentiment
def color_sentiment(val: str) -> str:
    return f"color: {'green' if val == 'Positive' else 'red'}"

# ------------------------
# Fetch and Process News
# ------------------------
try:
    response = requests.get(URL, timeout=10)
    data = response.json()
    articles = data.get("articles", [])
except Exception as e:
    st.error(f"Error fetching news: {e}")
    articles = []

if articles:
    df = pd.DataFrame(articles)

    # Ensure description column exists
    if "description" not in df.columns:
        df["description"] = ""

    # Combine title + description
    df["full_text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    df["sentiment"] = df["full_text"].apply(get_sentiment)

    # Display styled dataframe
    st.subheader("Latest News with Sentiment")
    styled_df = df[["title", "sentiment"]].style.applymap(color_sentiment, subset=["sentiment"])
    st.dataframe(styled_df, use_container_width=True)

    # Sentiment distribution
    st.subheader("Sentiment Distribution")
    st.bar_chart(df["sentiment"].value_counts())

else:
    st.warning("⚠️ No news fetched. Check your API key or internet connection.")

