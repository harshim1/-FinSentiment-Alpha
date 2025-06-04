# Flask API to serve sentiment scores
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "FinSentiment Alpha API is running!"

@app.route('/sentiment')
def sentiment():
    symbol = request.args.get("symbol", "AAPL")
    # Placeholder response
    return jsonify({"symbol": symbol, "sentiment": "positive"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
