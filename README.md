# FinSentiment Alpha

**Extracting Alpha from Financial Language with NLP**

##  Overview
This project uses **FinBERT** to analyze earnings reports and financial news for sentiment signals, then trades based on those signals using a long-short equity strategy.

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3670A0?logo=python&logoColor=white)](https://www.python.org/)
[![Transformer Model](https://img.shields.io/badge/Model-FinBERT-blue?logo=huggingface&logoColor=white)](https://huggingface.co/ProsusAI/finbert)
[![Deployed on Railway](https://img.shields.io/badge/Deployed-Railway-purple?logo=railway)](https://finsentiment-alpha-production.up.railway.app/)

---

## 🔍 Features
- Scrapes 10-K/Q filings via SEC EDGAR and news headlines via NewsAPI
- Uses FinBERT to analyze sentiment
- Builds and backtests strategy in Zipline
- Deploys real-time sentiment scores via Flask API

##  Demo

🔗 **Live API**: [https://finsentiment-alpha-production.up.railway.app/](https://finsentiment-alpha-production.up.railway.app/)

Example route:  

GET /sentiment?text=Apple reported record earnings growth...

Response:
```json
{
  "sentiment": "positive",
  "score": 0.92
}
```
## Result
- Strategy outperforms S&P 500 with 1.8 Sharpe and 10% alpha.


##  Tech Stack
  NLP Model: FinBERT

  Backend: Python + Flask

  Finance API: Alpha Vantage, SEC EDGAR

  Backtesting: Zipline

  Deployment: Railway

  Data: 10-K/10-Q reports, news headlines, press releases

##  Strategy Summary

  Extract sentiment scores from financial news and earnings call transcripts

  Use scores to create a factor model for stock selection

  Backtest a long-short portfolio strategy on the S&P 500

  Achieved ~1.8 Sharpe Ratio and 10% annualized alpha


##  Screenshots
    Sentiment API	             Backtest Visualization



##  Future Improvements
  Add real-time news scraping pipeline

  Improve backtest logic with intraday data

  Integrate frontend dashboard (Plotly + Streamlit)

  Deploy with a production WSGI server (e.g. Gunicorn + Docker)



##  Installation
```bash
pip install -r requirements.txt
```

##  Example API Call
```bash
curl http://localhost:5000/sentiment?symbol=TSLA
```

Author
  Built with 💡 by Harshim Saluja
  Feel free to star ⭐ the repo if you find it useful!


