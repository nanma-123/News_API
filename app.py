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
nltk.download('vader_lexicon')

# ------------------------
# Spark session setup
# ------------------------
@st.cache_resource
def init_spark():
    return SparkSession.builder \
        .appName("RealTimeNewsSentiment") \
        .master("local[*]") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

spark_session = init_spark()

# ------------------------
# API Key
# ------------------------
NEWSAPI_KEY = "NEWS_API_KEY"

# ------------------------
# Fetch news function
# ------------------------
def get_news_data(query, limit=5):
    endpoint = f"https://newsapi.org/v2/everything?q={query}&pageSize={limit}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    response = requests.get(endpoint)
    articles = response.json().get("articles", [])
    return pd.DataFrame([{"headline": a["title"]} for a in articles if a.get("title")])

# ------------------------
# Sentiment labeling
# ------------------------
def assign_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    
    def sentiment_label(text):
        score = analyzer.polarity_scores(text)["compound"]
        if score > 0.05:
            return "Positive"
        elif score < -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    df["sentiment_text"] = df["headline"].apply(sentiment_label)
    label_dict = {"Positive": 2.0, "Neutral": 1.0, "Negative": 0.0}
    df["sentiment_label"] = df["sentiment_text"].map(label_dict)
    return df

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="🗞 Real-Time News Sentiment", page_icon="📰", layout="wide")
st.header("📰 Real-Time News Sentiment Dashboard")

search_term = st.text_input("Enter Topic", "technology")
operation_mode = st.radio("Select Mode", ["Train Model", "Predict Sentiment"])
article_count = st.slider("Number of Articles", 3, 20, 5)

if st.button("Execute"):
    news_df = get_news_data(search_term, article_count)
    
    if news_df.empty:
        st.warning("No articles found for this topic.")
    else:
        if operation_mode == "Train Model":
            # Label and prepare data
            news_df = assign_sentiment(news_df)
            spark_df = spark_session.createDataFrame(news_df)
            
            # Spark ML pipeline
            text_tokenizer = Tokenizer(inputCol="headline", outputCol="tokens")
            tokenized_data = text_tokenizer.transform(spark_df)
            
            hash_vectorizer = HashingTF(inputCol="tokens", outputCol="rawFeatures")
            features_data = hash_vectorizer.transform(tokenized_data)
            
            idf_transformer = IDF(inputCol="rawFeatures", outputCol="features")
            idf_model = idf_transformer.fit(features_data)
            final_data = idf_model.transform(features_data)
            
            classifier = LogisticRegression(maxIter=10, regParam=0.001)
            trained_model = classifier.fit(final_data)
            
            evaluator = MulticlassClassificationEvaluator(
                labelCol="sentiment_label", predictionCol="prediction", metricName="accuracy")
            training_acc = evaluator.evaluate(trained_model.transform(final_data))
            
            # Store in session
            st.session_state.update({
                "trained_model": trained_model,
                "text_tokenizer": text_tokenizer,
                "hash_vectorizer": hash_vectorizer,
                "idf_model": idf_model
            })
            
            st.success("✅ Model trained successfully!")
            st.info(f"Training Accuracy: {training_acc:.2%}")
            st.subheader("Training Data Sample")
            st.dataframe(news_df[["headline", "sentiment_text"]])
        
        elif operation_mode == "Predict Sentiment":
            if "trained_model" not in st.session_state:
                st.error("Please train the model first!")
            else:
                spark_df = spark_session.createDataFrame(news_df)
                
                # Load pipeline from session
                tokenizer = st.session_state["text_tokenizer"]
                hasher = st.session_state["hash_vectorizer"]
                idf_model = st.session_state["idf_model"]
                model = st.session_state["trained_model"]
                
                tokenized_data = tokenizer.transform(spark_df)
                hashed_data = hasher.transform(tokenized_data)
                final_features = idf_model.transform(hashed_data)
                
                predictions = model.transform(final_features)
                pred_df = predictions.select("headline", "prediction").toPandas()
                reverse_map = {2.0: "Positive", 1.0: "Neutral", 0.0: "Negative"}
                pred_df["predicted_sentiment"] = pred_df["prediction"].map(reverse_map)
                
                st.subheader("Predictions")
                st.dataframe(pred_df[["headline", "predicted_sentiment"]])
                
                st.subheader("Simulated Real-Time Feed")
                for idx, row in pred_df.iterrows():
                    st.markdown(f"**{row['headline']}** → {row['predicted_sentiment']}")
                    time.sleep(0.5)
                
                # Sentiment distribution chart
                chart_data = pred_df.groupby("predicted_sentiment").size().reset_index(name="count")
                sentiment_chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                    x='predicted_sentiment',
                    y='count',
                    color='predicted_sentiment'
                ).properties(height=350, width=500)
                
                st.altair_chart(sentiment_chart, use_container_width=True)
