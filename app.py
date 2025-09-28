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

nltk.download("vader_lexicon", quiet=True)

# Cache Spark session
@st.cache_resource
def init_spark():
    return SparkSession.builder \
        .appName("NewsSentimentApp") \
        .master("local[*]") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

spark = init_spark()

# API Key (replace with st.secrets in production)
API_KEY = "e35c3d77cbcb402eaadcafbf5bcedac4"

# ------------------------------
# Helper Functions
# ------------------------------
def get_news(query, limit=5):
    """Fetch latest news headlines from News API"""
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize={limit}&sortBy=publishedAt&apiKey={API_KEY}"
    resp = requests.get(url).json()
    articles = resp.get("articles", [])
    return pd.DataFrame([{"headline": a["title"]} for a in articles if a.get("title")])

def add_sentiment(df):
    """Classify sentiment using VADER and add labels"""
    sia = SentimentIntensityAnalyzer()
    def classify(text):
        score = sia.polarity_scores(text)["compound"]
        return "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
    df["sentiment"] = df["headline"].apply(classify)
    df["label"] = df["sentiment"].map({"Positive": 2.0, "Neutral": 1.0, "Negative": 0.0})
    return df

def spark_pipeline(df, mode="train"):
    """Train or predict sentiment using PySpark pipeline"""
    if mode == "train":
        df = add_sentiment(df)

    spark_df = spark.createDataFrame(df)
    tok = Tokenizer(inputCol="headline", outputCol="tokens")
    hash_tf = HashingTF(inputCol="tokens", outputCol="tf")
    idf = IDF(inputCol="tf", outputCol="features")

    # Tokenize & transform
    tokenized = tok.transform(spark_df)
    tf_data = hash_tf.transform(tokenized)
    final_data = idf.fit(tf_data).transform(tf_data) if mode == "train" else st.session_state["idf"].transform(tf_data)

    if mode == "train":
        lr = LogisticRegression(maxIter=10, regParam=0.001)
        model = lr.fit(final_data)
        acc = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        ).evaluate(model.transform(final_data))
        return model, tok, hash_tf, idf.fit(tf_data), acc, df
    else:
        model = st.session_state["model"]
        preds = model.transform(final_data).select("headline", "prediction").toPandas()
        preds["sentiment"] = preds["prediction"].map({2.0: "Positive", 1.0: "Neutral", 0.0: "Negative"})
        return preds

# Streamlit Dashboard

st.set_page_config(page_title="Global News Sentiment", layout="wide")
st.title("Real-Time Global News Sentiment Dashboard")

# Sidebar
st.sidebar.header("🔧 Controls")
query = st.sidebar.text_input("Enter Topic", "Climate Change")
mode = st.sidebar.radio("Choose Mode", ["Train Model", "Predict Sentiment"])
limit = st.sidebar.slider("Number of Articles", 5, 25, 10)

if st.sidebar.button("Run Analysis"):
    news_df = get_news(query, limit)
    if news_df.empty:
        st.warning("No news articles found.")
    else:
        if mode == "Train Model":
            model, tok, hash_tf, idf, acc, train_df = spark_pipeline(news_df, mode="train")
            st.session_state.update({"model": model, "tokenizer": tok, "hashing": hash_tf, "idf": idf})

            st.success(f"Model trained! Accuracy: {acc:.2%}")
            st.subheader("Training Data with VADER Labels")
            st.dataframe(train_df[["headline", "sentiment"]])

        elif mode == "Predict Sentiment":
            if "model" not in st.session_state:
                st.error("Please train the model first!")
            else:
                preds = spark_pipeline(news_df, mode="predict")

                st.subheader("Predicted Sentiments")
                for _, row in preds.iterrows():
                    st.markdown(f"- **{row['headline']}** →  {row['sentiment']}")
                    time.sleep(0.2)

                # Distribution Chart
                st.subheader("Sentiment Distribution")
                chart_data = preds.groupby("sentiment").size().reset_index(name="count")
                chart = alt.Chart(chart_data).mark_arc(innerRadius=50).encode(
                    theta="count", color="sentiment", tooltip=["sentiment", "count"]
                )
                st.altair_chart(chart, use_container_width=True)
