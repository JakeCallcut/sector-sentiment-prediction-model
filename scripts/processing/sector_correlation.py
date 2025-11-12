import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr
from pathlib import Path

#GLOBALS
SENT_PATH = "../../data/clean_Twitter/daily_sentiment_finbert.csv"  # expects columns: market_date,date?, sentiment_mean, n_tweets
OUT_DIR = Path("../../data/analysis/sectors/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2020-01-01"
END = "2020-12-31"

sector_to_ticker = {
    "Communication_Services": "XLC",
    "Consumer_Discretionary": "XLY",
    "Consumer_Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real_Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
    "All_Sectors": "SPY",              
    "Gold_Index_Fund": "GLD",          
    "United_States_Oil_Fund": "USO",
}

sent = pd.read_csv(SENT_PATH)
if "date" not in sent.columns and "market_date" in sent.columns:
    sent = sent.rename(columns={"market_date": "date"})
sent["date"] = pd.to_datetime(sent["date"]).dt.date
sent = sent[["date", "sentiment_mean", "n_tweets"]]

# HELPER FUNCTIONS
def get_returns(ticker: str) -> pd.DataFrame:
    """
    Fetch daily close, compute simple returns.
    Uses Ticker().history to avoid MultiIndex columns from yf.download.
    """
    hist = yf.Ticker(ticker).history(start=START, end=END, auto_adjust=True)
    px = (
        hist.reset_index()[["Date", "Close"]]
        .rename(columns={"Date": "date", "Close": "close"})
    )
    px["date"] = px["date"].dt.date
    px["return"] = px["close"].pct_change()
    return px[["date", "return"]]

# loop through all sectors
results = []

for sector, ticker in sector_to_ticker.items():
    print(f"Processing {sector} ({ticker}) ...")
    px = get_returns(ticker)

    # one sentiment series applied to all sectors
    merged = (
        pd.merge(px, sent, on="date", how="inner")
          .dropna(subset=["return", "sentiment_mean"])
    )

    if len(merged) < 10:
        print(f"  ⚠️ Skipping {sector} — not enough overlapping days")
        continue

    #PEARSON CORRELATION COEFFICIENTS
    r, p = pearsonr(merged["sentiment_mean"], merged["return"])

    #ADDRESSING LAG-1
    merged = merged.sort_values("date")
    merged["sentiment_lag1"] = merged["sentiment_mean"].shift(1)
    lag_df = merged.dropna(subset=["sentiment_lag1", "return"])
    r_lag, p_lag = pearsonr(lag_df["sentiment_lag1"], lag_df["return"])

    out_path = OUT_DIR / f"{sector}_merged.csv"
    merged.to_csv(out_path, index=False)
    print(f"  Saved: {out_path.name} ({len(merged)} rows)")

    results.append({
        "sector": sector,
        "ticker": ticker,
        "same_day_r": round(r, 4),
        "same_day_p": round(p, 4),
        "lag1_r": round(r_lag, 4),
        "lag1_p": round(p_lag, 4),
        "n_obs": int(len(merged)),
    })

# SUMMARY TABLE
summary = pd.DataFrame(results).sort_values("sector")
summary_path = OUT_DIR / "sector_correlation_summary.csv"
summary.to_csv(summary_path, index=False)

print("\nSummary saved to", summary_path)
print(summary)