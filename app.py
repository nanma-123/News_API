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

# Download VADER lexicon
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

# API Key from secrets
API_KEY = "65af9d35def2491cbf609f0916e0dabf"

# Fetch latest news
def get_news(query, limit=5):
    url = f"https://newsapi.org/v2/everything?q={query}&pageSize={limit}&sortBy=publishedAt&apiKey={API_KEY}"
    resp = requests.get(url).json()
    articles = resp.get("articles", [])
    return pd.DataFrame([{"headline": a["title"]} for a in articles if a.get("title")])

# Sentiment with VADER
def add_sentiment(df):
    sia = SentimentIntensityAnalyzer()
    def classify(text):
        score = sia.polarity_scores(text)["compound"]
        return "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
    df["sentiment"] = df["headline"].apply(classify)
    df["label"] = df["sentiment"].map({"Positive": 2.0, "Neutral": 1.0, "Negative": 0.0})
    return df

# Streamlit UI
st.set_page_config(page_title="AI News Sentiment", page_icon="📰", layout="wide")
st.title("📰 News Sentiment Dashboard with PySpark")

query = st.text_input("Enter Topic", "Artificial Intelligence")
mode = st.radio("Choose Mode", ["Train Model", "Predict Sentiment"], horizontal=True)
limit = st.slider("Number of Articles", 3, 20, 8)

if st.button("Run Analysis"):
    news_df = get_news(query, limit)
    if news_df.empty:
        st.warning("No news articles found.")
    else:
        if mode == "Train Model":
            news_df = add_sentiment(news_df)
            spark_df = spark.createDataFrame(news_df)

            # Pipeline
            tokenizer = Tokenizer(inputCol="headline", outputCol="tokens")
            tokenized = tokenizer.transform(spark_df)

            hashing = HashingTF(inputCol="tokens", outputCol="tf")
            tf_data = hashing.transform(tokenized)

            idf = IDF(inputCol="tf", outputCol="features").fit(tf_data)
            final_data = idf.transform(tf_data)

            lr = LogisticRegression(maxIter=10, regParam=0.001)
            model = lr.fit(final_data)

            acc = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="accuracy"
            ).evaluate(model.transform(final_data))

            # Save in session
            st.session_state.update({
                "model": model, "tokenizer": tokenizer,
                "hashing": hashing, "idf": idf
            })

            st.success(f"Model trained successfully! ✅ Accuracy: {acc:.2%}")
            st.dataframe(news_df[["headline", "sentiment"]])

        elif mode == "Predict Sentiment":
            if "model" not in st.session_state:
                st.error("Please train the model first!")
            else:
                spark_df = spark.createDataFrame(news_df)
                tok = st.session_state["tokenizer"]
                hash_tf = st.session_state["hashing"]
                idf_model = st.session_state["idf"]
                model = st.session_state["model"]

                tokenized = tok.transform(spark_df)
                tf_data = hash_tf.transform(tokenized)
                final_data = idf_model.transform(tf_data)
                preds = model.transform(final_data).select("headline", "prediction").toPandas()

                reverse_map = {2.0: "Positive", 1.0: "Neutral", 0.0: "Negative"}
                preds["sentiment"] = preds["prediction"].map(reverse_map)

                # Display results
                st.subheader("Predicted Sentiments")
                for _, row in preds.iterrows():
                    st.markdown(f"✅ *{row['headline']}* → {row['sentiment']}")
                    time.sleep(0.5)

                # Chart
                st.subheader("Sentiment Distribution")
                chart_data = preds.groupby("sentiment").size().reset_index(name="count")
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X("sentiment", sort=["Positive", "Neutral", "Negative"]),
                    y="count",
                    color="sentiment"
                )
                st.altair_chart(chart, use_container_width=True)
