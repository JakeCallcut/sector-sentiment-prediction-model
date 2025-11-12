# finbert_score_filtered.py
# pip install pandas numpy torch transformers

import re
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

# ---------- paths ----------
INPUT = "../../data/clean_Twitter/clean_tweets_2020.csv"   # expects: market_date, text
TWEET_OUT = "../../data/clean_Twitter/tweets_with_finbert.csv"
DAILY_OUT = "../../data/clean_Twitter/daily_sentiment_finbert.csv"

# ---------- market-relevance filters ----------
CASHTAG_RE = re.compile(r"(?<!\\w)\\$[A-Za-z]{1,5}(?:\\.[A-Za-z]{1,2})?(?!\\w)")

TICKERS = [
    # broad indices & ETFs
    "SPY","SPX","NDX","QQQ","DIA","DJIA","IWM","VIX",
    # sector SPDRs
    "XLC","XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLRE","XLK","XLU",
    # commodities / common futures mnemonics
    "GLD","USO","CL","WTI","BRENT","GC","SI"
]
TICKER_RE = re.compile(r"(?<![A-Z0-9])(" + "|".join(TICKERS) + r")(?![A-Z0-9])")

SECTOR_TERMS = [
    "technology","tech","energy","financials","banks","banking","healthcare",
    "industrials","materials","real estate","utilities","communication services",
    "consumer staples","consumer discretionary","semis","semiconductors","oil","gold",
    "commodity","commodities"
]
FINANCE_TERMS = [
    "stocks","equities","equity","market","markets","risk-on","risk off","risk-off",
    "volatility","selloff","sell-off","rally","bear","bull","yield","yields","treasury",
    "bond","bonds","spread","spreads","credit","duration","curve","inversion","inverted",
    "earnings","guidance","buyback","dividend","valuation","multiple","eps","revenue",
    "outlook","upgrade","downgrade"
]
MACRO_TERMS = [
    "fomc","fed","powell","rate","rates","hike","cut","dot plot","qe","qt",
    "balance sheet","cpi","pce","inflation","deflation","gdp","payrolls","nonfarm",
    "nfp","ism","pmi","retail sales","jobless claims","initial claims"
]

SECTOR_RE  = re.compile("|".join(SECTOR_TERMS), re.IGNORECASE)
FINANCE_RE = re.compile("|".join(FINANCE_TERMS), re.IGNORECASE)
MACRO_RE   = re.compile("|".join(MACRO_TERMS), re.IGNORECASE)

def is_market_related(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    if CASHTAG_RE.search(text):
        return True
    if TICKER_RE.search(text):
        return True
    if SECTOR_RE.search(text):
        return True
    if FINANCE_RE.search(text):
        return True
    if MACRO_RE.search(text):
        return True
    return False

# ---------- load data ----------
df = pd.read_csv(INPUT)

# Keep only finance/macro-relevant tweets
mask = df["text"].apply(is_market_related)
df = df[mask].copy()
print(f"Filtered to market-relevant tweets: {len(df):,}")

# ---------- model ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ProsusAI/finbert"  # finance-news friendly
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
mdl.eval()

def score_batch(texts, max_len=128):
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**{k: v.to(device) for k, v in enc.items()})
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()  # [neg, neu, pos]
    return probs[:, 2] - probs[:, 0]  # pos - neg in [-1, 1]

# ---------- scoring ----------
BATCH = 512
texts = df["text"].fillna("").astype(str).tolist()
scores = np.empty(len(df), dtype=np.float32)

for i in range(0, len(texts), BATCH):
    batch = texts[i:i+BATCH]
    scores[i:i+len(batch)] = score_batch(batch)

df["finbert_score"] = scores

# ---------- daily aggregation ----------
# Assumes df has 'market_date' column (YYYY-MM-DD aligned to US/Eastern close in your earlier pipeline)
daily = (
    df.groupby("market_date", as_index=False)
      .agg(n_tweets=("text", "count"),
           sentiment_mean=("finbert_score", "mean"))
)

# ---------- save ----------
Path(TWEET_OUT).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(TWEET_OUT, index=False)
daily.to_csv(DAILY_OUT, index=False)

print(f"Tweet-level -> {TWEET_OUT}  ({len(df):,} rows)")
print(f"Daily      -> {DAILY_OUT}  ({len(daily):,} days)")