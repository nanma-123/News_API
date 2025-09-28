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

# Download VADER lexicon (only once)
nltk.download('vader_lexicon')

# Spark session caching
@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("NewsSentimentStreaming") \
        .master("local[*]") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

spark = get_spark()

# Get API key from secrets
NEWS_API_KEY ="e35c3d77cbcb402eaadcafbf5bcedac4"

# Fetch news from NewsAPI
def fetch_news(topic, page_size=5):
    url = f"https://newsapi.org/v2/everything?q={topic}&pageSize={page_size}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    r = requests.get(url)
    articles = r.json().get("articles", [])
    return pd.DataFrame([{"title": a["title"]} for a in articles if a.get("title")])

# Sentiment labeling with VADER
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
    # Map to numeric labels for Spark ML
    label_map = {"Positive": 2.0, "Neutral": 1.0, "Negative": 0.0}
    df_pd["label"] = df_pd["sentiment"].map(label_map)
    return df_pd

# Streamlit UI
st.title("📰 PySpark News Sentiment Dashboard")

topic = st.text_input("Topic", "technology")
mode = st.radio("Mode", ["Bootstrap", "Predict"])
num_articles = st.slider("Number of articles", 3, 20, 5)

if st.button("Run"):
    df_pd = fetch_news(topic, num_articles)
    if df_pd.empty:
        st.error("No news found.")
    else:
        if mode == "Bootstrap":
            # Label data
            df_pd = label_sentiment(df_pd)
            df_spark = spark.createDataFrame(df_pd)

            # Spark ML Pipeline
            tokenizer = Tokenizer(inputCol="title", outputCol="words")
            words_data = tokenizer.transform(df_spark)

            hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
            featurized_data = hashingTF.transform(words_data)

            idf = IDF(inputCol="rawFeatures", outputCol="features")
            idf_model = idf.fit(featurized_data)
            rescaled_data = idf_model.transform(featurized_data)

            lr = LogisticRegression(maxIter=10, regParam=0.001)
            model = lr.fit(rescaled_data)

            # Training accuracy
            evaluator = MulticlassClassificationEvaluator(
                labelCol="label", predictionCol="prediction", metricName="accuracy")
            train_accuracy = evaluator.evaluate(model.transform(rescaled_data))

            # Store in session
            st.session_state.update({
                "model": model,
                "tokenizer": tokenizer,
                "hashingTF": hashingTF,
                "idf_model": idf_model
            })

            st.success("Model trained successfully!")
            st.info(f"Training Accuracy: {train_accuracy:.2%}")
            st.subheader("Bootstrap Data")
            st.write(df_pd[["title", "sentiment"]])

        elif mode == "Predict":
            if "model" not in st.session_state:
                st.error("Run Bootstrap first!")
            else:
                # Convert new data to Spark DataFrame
                df_spark = spark.createDataFrame(df_pd)

                # Transform data using cached pipeline
                tokenizer = st.session_state["tokenizer"]
                hashingTF = st.session_state["hashingTF"]
                idf_model = st.session_state["idf_model"]
                model = st.session_state["model"]

                words_data = tokenizer.transform(df_spark)
                featurized_data = hashingTF.transform(words_data)
                rescaled_data = idf_model.transform(featurized_data)
                predictions = model.transform(rescaled_data)

                predictions_df = predictions.select("title", "prediction").toPandas()
                label_map_reverse = {2.0: "Positive", 1.0: "Neutral", 0.0: "Negative"}
                predictions_df["sentiment"] = predictions_df["prediction"].map(label_map_reverse)

                st.subheader("Predictions")
                st.write(predictions_df[["title", "sentiment"]])

                # Real-time streaming simulation (display each row with delay)
                st.subheader("Streaming Simulation")
                for idx, row in predictions_df.iterrows():
                    st.write(f"**{row['title']}** → {row['sentiment']}")
                    time.sleep(0.5)

                # Altair bar chart for sentiment distribution
                chart_data = predictions_df.groupby("sentiment").size().reset_index(name='count')
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x='sentiment',
                    y='count',
                    color='sentiment'
                )
                st.altair_chart(chart, use_container_width=True)
