"""
FinSentiment Alpha — Full Pipeline
Harshim Saluja | Penn State CDS&S

Runs end-to-end:
  1. Pull 10-K/10-Q filings from SEC EDGAR (free)
  2. Score MD&A sections with FinBERT
  3. Build long-short equity portfolio
  4. Walk-forward OOS validation + stats for paper

Usage:
  pip install transformers torch yfinance requests pandas numpy scipy
  python finsentiment_pipeline.py

Runtime: ~60-90 min on M-series Mac (EDGAR rate limit is the bottleneck)
"""

import os, time, json, re, warnings
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
warnings.filterwarnings('ignore')

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Adjust these to match your original setup
UNIVERSE = [                        # S&P 100 subset — runs faster than full 500
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","BRK-B","JPM","V",
    "UNH","JNJ","XOM","PG","MA","HD","CVX","MRK","ABBV","PEP",
    "KO","COST","AVGO","WMT","LLY","TMO","MCD","ACN","ABT","BAC",
    "CRM","NKE","ORCL","TXN","DHR","NEE","PM","LIN","AMGN","RTX",
    "HON","UPS","LOW","SBUX","QCOM","IBM","INTU","AMD","CAT","GS",
    "AXP","SPGI","BLK","ISRG","GILD","ADI","NOW","REGN","MDLZ","MMC",
    "ZTS","MO","VRTX","EL","CME","LRCX","CI","SYK","C","DE",
    "CB","AON","PYPL","DUK","SO","D","EMR","PLD","CCI","WM",
    "EW","NSC","ITW","MCO","ICE","FDX","APD","ECL","HUM","BSX",
    "DXCM","IQV","PSA","SHW","MSI","KLAC","FTNT","CTAS","MRNA","SNPS",
]
START_DATE   = "2019-01-01"
END_DATE     = "2023-12-31"
SPLIT_DATE   = "2022-01-01"        # OOS starts here (~70/30 split)
TC_BPS       = 2                   # transaction cost per leg in basis points
LAG_DAYS     = 3                   # days after filing before trading
MAX_FILINGS  = 3                   # filings per ticker (keeps runtime short)
EDGAR_SLEEP  = 0.12                # stay under 10 req/sec EDGAR limit

HEADERS = {"User-Agent": "Harshim Saluja harshimsaluja1@gmail.com"}  # EDGAR requires this


# ── STEP 1: GET CIK NUMBERS FROM EDGAR ───────────────────────────────────────
def get_cik(ticker):
    """Map ticker to SEC CIK number."""
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2019-01-01&enddt=2019-12-31&forms=10-K"
    try:
        r = requests.get(
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=&CIK={ticker}&type=10-K&dateb=&owner=include&count=1&search_text=&output=atom",
            headers=HEADERS, timeout=10
        )
        # Parse CIK from atom feed
        match = re.search(r'<cik>(\d+)</cik>', r.text)
        if match:
            return match.group(1).zfill(10)
    except:
        pass
    return None


def get_filings(cik, form_type="10-K", start=START_DATE, end=END_DATE):
    """Get filing metadata from EDGAR XBRL API."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        data = r.json()
        filings = data.get('filings', {}).get('recent', {})
        df = pd.DataFrame({
            'form':        filings.get('form', []),
            'filed':       filings.get('filingDate', []),
            'accession':   filings.get('accessionNumber', []),
            'primaryDoc':  filings.get('primaryDocument', []),
        })
        df = df[df['form'].isin(['10-K', '10-Q'])]
        df['filed'] = pd.to_datetime(df['filed'])
        df = df[(df['filed'] >= start) & (df['filed'] <= end)]
        return df.head(MAX_FILINGS)
    except Exception as e:
        return pd.DataFrame()


def get_mda_text(cik, accession, primary_doc):
    """Fetch and extract MD&A section from filing."""
    acc_clean = accession.replace('-', '')
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{primary_doc}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        text = r.text

        # Strip HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&nbsp;|&amp;|&lt;|&gt;', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Extract MD&A section (Item 7)
        mda_pattern = re.compile(
            r'(?i)item\s*7[a\.]?\s*\.?\s*management.{0,50}discussion',
        )
        match = mda_pattern.search(text)
        if match:
            start = match.start()
            # Take next 5000 chars (FinBERT chunking handles length)
            mda = text[start:start + 5000]
            return mda
        else:
            # Fallback: use first 3000 chars of body
            return text[:3000]
    except:
        return ""


# ── STEP 2: FINBERT SCORING ───────────────────────────────────────────────────
def load_finbert():
    """Load FinBERT model. Downloads ~440MB on first run."""
    print("Loading FinBERT (downloads ~440MB if first time)...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def score_text(text, tokenizer, model, chunk_size=400, stride=50):
    """
    Score text with FinBERT using overlapping chunks.
    Returns net sentiment: P(positive) - P(negative)
    """
    import torch
    import torch.nn.functional as F

    if not text or len(text.strip()) < 50:
        return 0.0

    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks, weights = [], []

    for i in range(0, max(1, len(tokens)), chunk_size - stride):
        chunk = tokens[i:i + chunk_size]
        if len(chunk) < 10:
            continue
        chunks.append(chunk)
        weights.append(len(chunk))

    if not chunks:
        return 0.0

    scores = []
    for chunk in chunks:
        inputs = tokenizer.prepare_for_model(
            chunk, return_tensors='pt',
            max_length=512, truncation=True
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0]
        # FinBERT labels: 0=positive, 1=negative, 2=neutral
        net = probs[0].item() - probs[1].item()
        scores.append(net)

    return float(np.average(scores, weights=weights[:len(scores)]))


# ── STEP 3: BUILD SIGNAL DATAFRAME ───────────────────────────────────────────
def build_signals(universe, tokenizer, model):
    """
    For each ticker: get filings, score MD&A, record
    (ticker, filing_date, trade_date, sentiment_score).
    """
    records = []
    total = len(universe)

    for i, ticker in enumerate(universe):
        print(f"  [{i+1}/{total}] {ticker}", end=" ", flush=True)

        cik = get_cik(ticker)
        if not cik:
            print("→ no CIK")
            continue
        time.sleep(EDGAR_SLEEP)

        filings = get_filings(cik)
        if filings.empty:
            print("→ no filings")
            continue

        for _, row in filings.iterrows():
            time.sleep(EDGAR_SLEEP)
            mda = get_mda_text(cik, row['accession'], row['primaryDoc'])
            if not mda:
                continue
            score = score_text(mda, tokenizer, model)
            trade_date = row['filed'] + timedelta(days=LAG_DAYS)  # 3-day lag
            records.append({
                'ticker':     ticker,
                'filed':      row['filed'],
                'trade_date': trade_date,
                'form':       row['form'],
                'sentiment':  score,
            })
            print(".", end="", flush=True)
        print()

    df = pd.DataFrame(records)
    df.to_csv('signals.csv', index=False)
    print(f"\nSaved {len(df)} signal observations → signals.csv")
    return df


# ── STEP 4: GET FORWARD RETURNS ───────────────────────────────────────────────
def get_returns(universe, start=START_DATE, end=END_DATE):
    """Download adjusted closing prices via yfinance."""
    import yfinance as yf
    print("\nDownloading price data from Yahoo Finance...")
    raw = yf.download(universe, start=start, end=end,
                      auto_adjust=True, progress=False)['Close']
    returns = raw.pct_change().dropna(how='all')
    returns.to_csv('returns.csv')
    print(f"Price data: {returns.shape[0]} days × {returns.shape[1]} tickers")
    return returns


def get_forward_returns(returns, holding_days=21):
    """21-trading-day forward return for each ticker on each date."""
    fwd = {}
    for col in returns.columns:
        fwd[col] = returns[col].rolling(holding_days).sum().shift(-holding_days)
    return pd.DataFrame(fwd)


# ── STEP 5: PORTFOLIO CONSTRUCTION ───────────────────────────────────────────
def build_portfolio(signals, returns):
    """
    On each trade_date, rank tickers by sentiment CHANGE,
    go long top decile, short bottom decile.
    Returns daily portfolio returns Series.
    """
    fwd_returns = get_forward_returns(returns)
    signals = signals.sort_values('trade_date')

    # Compute sentiment change (current - previous filing per ticker)
    signals['sentiment_lag'] = signals.groupby('ticker')['sentiment'].shift(1)
    signals['sentiment_change'] = signals['sentiment'] - signals['sentiment_lag']
    signals = signals.dropna(subset=['sentiment_change'])

    # Snap trade dates to valid market dates
    valid_dates = returns.index
    signals['trade_date'] = pd.to_datetime(signals['trade_date'])
    signals['trade_date'] = signals['trade_date'].apply(
        lambda d: valid_dates[valid_dates >= d][0]
        if len(valid_dates[valid_dates >= d]) > 0 else None
    )
    signals = signals.dropna(subset=['trade_date'])

    portfolio_returns = []

    for trade_date, group in signals.groupby('trade_date'):
        if len(group) < 6:
            continue

        # Rank by sentiment change
        group = group.set_index('ticker')
        n = len(group)
        decile = max(1, n // 10)

        long_tickers  = group['sentiment_change'].nlargest(decile).index.tolist()
        short_tickers = group['sentiment_change'].nsmallest(decile).index.tolist()

        # Get 21-day forward return window
        if trade_date not in fwd_returns.index:
            continue

        long_ret  = fwd_returns.loc[trade_date, [t for t in long_tickers  if t in fwd_returns.columns]].mean()
        short_ret = fwd_returns.loc[trade_date, [t for t in short_tickers if t in fwd_returns.columns]].mean()

        if pd.isna(long_ret) or pd.isna(short_ret):
            continue

        # Apply TC: 2bps each leg
        tc = TC_BPS / 10000
        ls_return = (long_ret - short_ret) - (2 * tc)

        portfolio_returns.append({
            'date':       trade_date,
            'ls_return':  ls_return,
            'long_ret':   long_ret,
            'short_ret':  short_ret,
            'n_long':     len(long_tickers),
            'n_short':    len(short_tickers),
        })

    port = pd.DataFrame(portfolio_returns).set_index('date').sort_index()
    port.to_csv('portfolio_returns.csv')
    return port


# ── STEP 6: OOS STATS ────────────────────────────────────────────────────────
def compute_stats(returns_series, label):
    r = returns_series.dropna()
    if len(r) < 5:
        print(f"{label}: insufficient data ({len(r)} obs)")
        return {}

    # Annualize (each obs = ~21 trading days, ~12 obs/yr)
    obs_per_year = 252 / 21
    ann_ret = r.mean() * obs_per_year
    ann_vol = r.std() * np.sqrt(obs_per_year)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

    cumret = (1 + r).cumprod()
    maxdd  = ((cumret - cumret.cummax()) / cumret.cummax()).min()
    calmar = ann_ret / abs(maxdd) if maxdd != 0 else np.nan

    # t-test: is mean return > 0?
    t_stat, p_val = stats.ttest_1samp(r, 0)

    print(f"\n{'='*50}")
    print(f"  {label}  (n={len(r)} rebalancing periods)")
    print(f"{'='*50}")
    print(f"  Ann. Return:      {ann_ret:>8.2%}")
    print(f"  Ann. Volatility:  {ann_vol:>8.2%}")
    print(f"  Sharpe Ratio:     {sharpe:>8.3f}  ← headline number")
    print(f"  Max Drawdown:     {maxdd:>8.2%}")
    print(f"  Calmar Ratio:     {calmar:>8.3f}")
    print(f"  t-stat (ret>0):   {t_stat:>8.3f}  (p={p_val:.3f})")
    print(f"{'='*50}")

    return {
        'label': label, 'n': len(r),
        'ann_return': ann_ret, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'max_drawdown': maxdd,
        'calmar': calmar, 't_stat': t_stat, 'p_value': p_val
    }


def walk_forward_validation(port, n_folds=4):
    """Walk-forward OOS validation."""
    print("\n\n=== WALK-FORWARD VALIDATION ===")
    r = port['ls_return'].dropna()
    fold_size = len(r) // (n_folds + 1)
    oos_all = []

    for i in range(1, n_folds + 1):
        train = r.iloc[:i * fold_size]
        test  = r.iloc[i * fold_size:(i+1) * fold_size]
        if len(test) < 3:
            continue
        sharpe_fold = (test.mean() / test.std()) * np.sqrt(252/21) if test.std() > 0 else np.nan
        print(f"  Fold {i}: train={train.index[0].date()}→{train.index[-1].date()} | "
              f"test n={len(test)} | OOS Sharpe={sharpe_fold:.3f}")
        oos_all.append(test)

    if oos_all:
        oos_combined = pd.concat(oos_all)
        wf_stats = compute_stats(oos_combined, "WALK-FORWARD OOS (all folds combined)")
        return wf_stats
    return {}


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FinSentiment Alpha Pipeline")
    print(f"  Universe: {len(UNIVERSE)} tickers | {START_DATE} → {END_DATE}")
    print(f"  OOS split: {SPLIT_DATE} | TC: {TC_BPS}bps/leg | Lag: {LAG_DAYS}d")
    print("=" * 60)

    # Load cached signals if available (skip re-scraping)
    if os.path.exists('signals.csv'):
        print("\nFound signals.csv — skipping EDGAR scrape.")
        print("Delete signals.csv to re-scrape from scratch.")
        signals = pd.read_csv('signals.csv', parse_dates=['filed', 'trade_date'])
    else:
        tokenizer, model = load_finbert()
        print(f"\nStep 1/3: Scraping {len(UNIVERSE)} tickers from SEC EDGAR...")
        signals = build_signals(UNIVERSE, tokenizer, model)

    print(f"\nSignal observations: {len(signals)}")
    print(signals[['ticker','filed','form','sentiment']].head(10).to_string())

    # Price data
    if os.path.exists('returns.csv'):
        print("\nFound returns.csv — skipping price download.")
        returns = pd.read_csv('returns.csv', index_col=0, parse_dates=True)
    else:
        returns = get_returns(UNIVERSE)

    # Portfolio
    print("\nStep 2/3: Building long-short portfolio...")
    port = build_portfolio(signals, returns)
    print(f"Portfolio observations: {len(port)}")

    # Stats
    print("\nStep 3/3: Computing performance statistics...")
    split = pd.Timestamp(SPLIT_DATE)

    is_port  = port[port.index <  split]['ls_return']
    oos_port = port[port.index >= split]['ls_return']

    is_stats  = compute_stats(is_port,  "IN-SAMPLE")
    oos_stats = compute_stats(oos_port, "OUT-OF-SAMPLE (simple holdout)")
    wf_stats  = walk_forward_validation(port)

    # Summary for paper
    print("\n\n" + "★" * 60)
    print("  NUMBERS FOR YOUR PAPER")
    print("★" * 60)
    if oos_stats:
        print(f"\n  OOS Sharpe (holdout):       {oos_stats.get('sharpe', 'N/A'):.3f}")
        print(f"  OOS Ann. Return:            {oos_stats.get('ann_return', 'N/A'):.2%}")
        print(f"  OOS Max Drawdown:           {oos_stats.get('max_drawdown', 'N/A'):.2%}")
        print(f"  OOS Calmar:                 {oos_stats.get('calmar', 'N/A'):.3f}")
        print(f"  t-stat (return > 0):        {oos_stats.get('t_stat', 'N/A'):.3f}")
    if wf_stats:
        print(f"\n  Walk-Forward OOS Sharpe:    {wf_stats.get('sharpe', 'N/A'):.3f}  ← use this in abstract")
    if is_stats and oos_stats:
        ratio = oos_stats.get('sharpe', 0) / is_stats.get('sharpe', 1)
        print(f"\n  IS/OOS Sharpe degradation:  {ratio:.2f}x")
        print(f"  (0.5–0.8 = healthy | >0.9 = suspicious | <0.3 = broken)")

    print("\n  → Fill these into finsentiment_alpha_ssrn_v2.tex")
    print("  → Then rerun pdflatex twice to rebuild the PDF")
    print("★" * 60)

    # Save summary
    summary = {**is_stats, **{f'oos_{k}': v for k, v in oos_stats.items()}}
    with open('stats_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nFull stats saved → stats_summary.json")


if __name__ == "__main__":
    main()
