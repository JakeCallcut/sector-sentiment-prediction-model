import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path

INPUT = "../../data/clean_Twitter/clean_tweets_2020.csv"
TWEET_OUT = "../../data/clean_Twitter/tweets_with_finbert.csv"
DAILY_OUT = "../../data/clean_Twitter/daily_sentiment_finbert.csv"

df = pd.read_csv(INPUT)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
mdl = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone").to(device)
mdl.eval()

def score_batch(texts, max_len=128):
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        out = mdl(**{k: v.to(device) for k, v in enc.items()})
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()  # [neg, neu, pos]
    return probs[:, 2] - probs[:, 0]  # continuous score in [-1, 1]

BATCH = 512
texts = df["text"].fillna("").astype(str).tolist()
scores = np.empty(len(df), dtype=np.float32)

for i in range(0, len(texts), BATCH):
    scores[i:i+BATCH] = score_batch(texts[i:i+BATCH])

df["finbert_score"] = scores

daily = (
    df.groupby("market_date", as_index=False)
      .agg(n_tweets=("text","count"),
           sentiment_mean=("finbert_score","mean"))
)

Path(TWEET_OUT).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(TWEET_OUT, index=False)
daily.to_csv(DAILY_OUT, index=False)

print(f"Tweet-level -> {TWEET_OUT}  ({len(df):,} rows)")
print(f"Daily      -> {DAILY_OUT}  ({len(daily):,} days)")

