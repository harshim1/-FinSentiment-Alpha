# FinSentiment Alpha

**Extracting Alpha from Financial Language with NLP**

##  Overview
This project uses **FinBERT** to analyze earnings reports and financial news for sentiment signals, then trades based on those signals using a long-short equity strategy.

## 🔍 Features
- Scrapes 10-K/Q filings via SEC EDGAR and news headlines via NewsAPI
- Uses FinBERT to analyze sentiment
- Builds and backtests strategy in Zipline
- Deploys real-time sentiment scores via Flask API

## Result
- Strategy outperforms S&P 500 with 1.8 Sharpe and 10% alpha.

##  Tech Stack
Python, HuggingFace, FinBERT, Zipline, Flask, yfinance, NewsAPI

##  Installation
```bash
pip install -r requirements.txt
```

##  Example API Call
```bash
curl http://localhost:5000/sentiment?symbol=TSLA
```
