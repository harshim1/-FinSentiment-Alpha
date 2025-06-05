# Flask API to serve sentiment scores
from flask import Flask, request, jsonify
from transformers import pipeline
import torch

app = Flask(__name__)

# Load FinBERT sentiment model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert"
)

@app.route("/")
def home():
    return "✅ FinSentiment Alpha is running. Try /sentiment?text=Tesla+beats+earnings"

@app.route("/sentiment", methods=["GET"])
def analyze_sentiment():
    text = request.args.get("text")
    if not text:
        return jsonify({"error": "No text provided."}), 400

    try:
        result = sentiment_pipeline(text)
        return jsonify({"text": text, "sentiment": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500