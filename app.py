import streamlit as st
import pandas as pd
import requests
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import altair as alt
import time

# --------------------------
# Setup
# --------------------------
nltk.download("vader_lexicon", quiet=True)

@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("NewsSentimentStreaming") \
        .master("local[*]") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

spark = get_spark()
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

# --------------------------
# Helpers
# --------------------------
def fetch_news(topic, page_size=5):
    """Fetch latest news headlines"""
    url = f"https://newsapi.org/v2/everything?q={topic}&pageSize={page_size}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    r = requests.get(url).json()
    articles = r.get("articles", [])
    return pd.DataFrame([{"title": a["title"]} for a in articles if a.get("title")])

def label_sentiment(df_pd):
    """Apply VADER sentiment + numeric mapping"""
    sia = SentimentIntensityAnalyzer()
    def classify(text):
        score = sia.polarity_scores(text)["compound"]
        return "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
    df_pd["sentiment"] = df_pd["title"].apply(classify)
    mapping = {"Positive": 2.0, "Neutral": 1.0, "Negative": 0.0}
    df_pd["label"] = df_pd["sentiment"].map(mapping)
    return df_pd

def train_model(df_pd):
    """Train logistic regression with Spark"""
    df_pd = label_sentiment(df_pd)
    df_spark = spark.createDataFrame(df_pd)

    tokenizer = Tokenizer(inputCol="title", outputCol="words")
    words = tokenizer.transform(df_spark)

    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
    featurized = hashingTF.transform(words)

    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idf_model = idf.fit(featurized)
    rescaled = idf_model.transform(featurized)

    lr = LogisticRegression(maxIter=10, regParam=0.001)
    model = lr.fit(rescaled)

    acc = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    ).evaluate(model.transform(rescaled))

    return model, tokenizer, hashingTF, idf_model, acc, df_pd

def predict_sentiment(df_pd):
    """Predict sentiment using cached model pipeline"""
    df_spark = spark.createDataFrame(df_pd)
    tok = st.session_state["tokenizer"]
    hashTF = st.session_state["hashingTF"]
    idf_model = st.session_state["idf_model"]
    model = st.session_state["model"]

    words = tok.transform(df_spark)
    feats = hashTF.transform(words)
    rescaled = idf_model.transform(feats)
    preds = model.transform(rescaled).select("title", "prediction").toPandas()

    rev_map = {2.0: "Positive", 1.0: "Neutral", 0.0: "Negative"}
    preds["sentiment"] = preds["prediction"].map(rev_map)
    return preds

# --------------------------
# Dashboard
# --------------------------
st.set_page_config(page_title="News Sentiment Dashboard", page_icon="📰", layout="wide")
st.title("Real-Time News Sentiment with PySpark")

# Sidebar
st.sidebar.header("🔧 Controls")
topic = st.sidebar.text_input("Enter Topic", "Climate Change")
mode = st.sidebar.radio("Mode", ["Bootstrap (Train)", "Predict"])
num_articles = st.sidebar.slider("Articles", 3, 20, 8)

if st.sidebar.button("🚀 Run"):
    df_pd = fetch_news(topic, num_articles)

    if df_pd.empty:
        st.error("⚠️ No news found.")
    else:
        if mode == "Bootstrap (Train)":
            model, tok, hashTF, idf_model, acc, df_pd = train_model(df_pd)

            st.session_state.update({
                "model": model, "tokenizer": tok,
                "hashingTF": hashTF, "idf_model": idf_model
            })

            st.success(f"Model trained successfully! Accuracy: {acc:.2%}")
            st.subheader("Training Data (with VADER Labels)")
            st.dataframe(df_pd[["title", "sentiment"]])

        elif mode == "Predict":
            if "model" not in st.session_state:
                st.error("Please bootstrap (train) first!")
            else:
                preds = predict_sentiment(df_pd)

                st.subheader("Predictions")
                for _, row in preds.iterrows():
                    st.markdown(f"- **{row['title']}** → 🎯 {row['sentiment']}")
                    time.sleep(0.3)

                # Sentiment distribution chart
                st.subheader("Sentiment Distribution")
                chart_data = preds.groupby("sentiment").size().reset_index(name="count")
                chart = alt.Chart(chart_data).mark_arc(innerRadius=50).encode(
                    theta="count", color="sentiment", tooltip=["sentiment", "count"]
                )
                st.altair_chart(chart, use_container_width=True)
