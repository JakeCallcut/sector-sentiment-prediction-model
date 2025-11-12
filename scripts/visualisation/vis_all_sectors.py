import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "grey",
    "grid.color": "lightgrey",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

IN_DIR = Path("../../data/analysis/sectors/")
FIG_DIR = IN_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = [
    "Communication_Services", "Consumer_Discretionary", "Consumer_Staples",
    "Energy", "Financials", "Healthcare", "Industrials",
    "Materials", "Real_Estate", "Technology", "Utilities",
    "All_Sectors", "Gold_Index_Fund", "United_States_Oil_Fund",
]

palette = sns.color_palette("husl", len(SECTORS))
sector_colors = dict(zip([s.replace("_", " ") for s in SECTORS], palette))

def _load_sector(sector):
    df = pd.read_csv(IN_DIR / f"{sector}_merged.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["sector"] = sector.replace("_", " ")
    keep = [c for c in ["date", "sector", "return", "sentiment_mean", "n_tweets", "sentiment_lag1"] if c in df.columns]
    return df[keep]

def load_all():
    frames, missing = [], []
    for s in SECTORS:
        try:
            frames.append(_load_sector(s))
        except FileNotFoundError:
            missing.append(s)
    if missing:
        print("Missing:", ", ".join(missing))
    return pd.concat(frames, ignore_index=True)

def zscore(x):
    sd = x.std(ddof=0)
    return (x - x.mean()) / (sd if sd and not np.isnan(sd) else 1)

# PLOTS 

def plot_correlation_bars(summary_csv=IN_DIR / "sector_correlation_summary.csv"):
    summary = pd.read_csv(summary_csv)
    labels = summary["sector"].str.replace("_", " ").tolist()
    x = np.arange(len(labels))
    width = 0.38
    colors = sns.color_palette("Set2", 2)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, summary["same_day_r"], width, label="Same-day r", color=colors[0])
    ax.bar(x + width/2, summary["lag1_r"], width, label="Lag-1 r", color=colors[1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_title("Single sentiment vs sector returns — correlations (2020)")
    ax.axhline(0, linewidth=0.8, color="grey")
    ax.legend()
    plt.tight_layout()
    out = FIG_DIR / "bar_correlations_by_sector.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

def plot_sentiment_over_all_returns(df):
    """
    Plot the single sentiment series over all sectors’ normalised cumulative returns.
    """
    ret_long = df[["date", "sector", "return"]].copy()
    ret_wide = ret_long.pivot(index="date", columns="sector", values="return").sort_index()
    cumret = (1 + ret_wide.fillna(0)).cumprod()
    cumret_z = cumret.apply(zscore)

    s = df[["date", "sentiment_mean"]].drop_duplicates(subset=["date"])
    s = s.set_index("date").reindex(cumret_z.index)["sentiment_mean"]
    s_z = zscore(s.fillna(method="ffill").fillna(0))

    plt.figure(figsize=(14, 7))
    for col in cumret_z.columns:
        plt.plot(
            cumret_z.index,
            cumret_z[col],
            linewidth=1.3,
            alpha=0.8,
            color=sector_colors.get(col, "light_grey"),
            label=col
        )

    # single sentiment line (dark grey)
    plt.plot(
        s_z.index, s_z.values,
        linewidth=2.2, color="black", label="Sentiment (z)"
    )

    plt.title("Single sentiment vs all sectors’ normalised cumulative returns (2020)")
    plt.xlabel("Date")
    plt.ylabel("Normalised value (z-score)")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "line_sentiment_over_sectors_returns.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

def plot_heatmap_single_sentiment(df):
    d = df[["date", "sector", "return", "sentiment_mean"]].copy()
    ret_w = d.pivot(index="date", columns="sector", values="return").sort_index()
    s = d[["date", "sentiment_mean"]].drop_duplicates(subset=["date"]).set_index("date").sort_index()["sentiment_mean"]
    s_lag1 = s.shift(1)
    sectors = ret_w.columns.tolist()
    mat = np.zeros((2, len(sectors))) * np.nan
    for j, sec in enumerate(sectors):
        r = ret_w[sec]
        valid_same = s.notna() & r.notna()
        valid_lag = s_lag1.notna() & r.notna()
        if valid_same.sum() >= 5:
            mat[0, j] = np.corrcoef(s[valid_same], r[valid_same])[0, 1]
        if valid_lag.sum() >= 5:
            mat[1, j] = np.corrcoef(s_lag1[valid_lag], r[valid_lag])[0, 1]
    fig, ax = plt.subplots(figsize=(14, 3.8))
    sns.heatmap(
        mat, ax=ax, cmap="coolwarm", center=0, cbar_kws={"label": "Pearson r"},
        xticklabels=sectors, yticklabels=["Same-day", "Lag-1"]
    )
    ax.set_xticklabels(sectors, rotation=45, ha="right")
    ax.set_title("Single sentiment vs sector returns — heatmap of Pearson r")
    plt.tight_layout()
    out = FIG_DIR / "heatmap_single_sentiment_vs_sectors.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

def plot_sector_dual_axis(df, sector_name="Technology"):
    sec = sector_name.replace("_", " ")
    d = df[df["sector"] == sec].copy().sort_values("date")
    if d.empty: return
    s_norm = zscore(d["sentiment_mean"])
    r_cum = (1 + d["return"].fillna(0)).cumprod()
    r_norm = zscore(r_cum)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(d["date"], s_norm, label="Sentiment (z)", color="grey", linewidth=1.5)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Sentiment (z)", color="grey")
    ax1.tick_params(axis='y', labelcolor="grey")

    ax2 = ax1.twinx()
    ax2.plot(d["date"], r_norm, label="Cumulative return (z)",
             color=sector_colors.get(sec, "steelblue"), linewidth=1.8)
    ax2.set_ylabel("Cumulative return (z)", color=sector_colors.get(sec, "steelblue"))
    ax2.tick_params(axis='y', labelcolor=sector_colors.get(sec, "steelblue"))

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=9)
    plt.title(f"{sec}: Sentiment vs Cumulative Returns (normalised)")
    plt.tight_layout()
    out = FIG_DIR / f"{sector_name}_dual_axis_sentiment_vs_returns.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

def plot_sector_lag_scatter(df, sector_name="Technology"):
    sec = sector_name.replace("_", " ")
    d = df[df["sector"] == sec].copy().sort_values("date")
    d["sentiment_lag1"] = d["sentiment_mean"].shift(1)
    d = d.dropna(subset=["sentiment_lag1", "return"])
    if len(d) < 5: return
    x, y = d["sentiment_lag1"].values, d["return"].values
    m, b = np.polyfit(x, y, 1)
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.6, color=sector_colors.get(sec, "darkslateblue"))
    plt.plot(x, m * x + b, color="black")
    plt.xlabel("Sentiment (t-1)")
    plt.ylabel("Return (t)")
    plt.title(f"{sec}: Lag-1 Sentiment vs Return")
    plt.tight_layout()
    out = FIG_DIR / f"{sector_name}_lag1_scatter.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

def plot_rolling_correlation(df, window=30, sector_name="Technology"):
    sec = sector_name.replace("_", " ")
    d = df[df["sector"] == sec].copy().sort_values("date")
    if d.empty: return
    s = d.set_index("date")["sentiment_mean"]
    r = d.set_index("date")["return"]
    roll = s.rolling(window).corr(r)
    plt.figure(figsize=(12, 4))
    plt.plot(roll.index, roll.values, color=sector_colors.get(sec, "teal"), linewidth=1.5)
    plt.axhline(0, linewidth=0.8, color="grey")
    plt.title(f"{sec}: {window}-day Rolling Correlation (Sentiment vs Return)")
    plt.xlabel("Date")
    plt.ylabel("Pearson r")
    plt.tight_layout()
    out = FIG_DIR / f"{sector_name}_rolling_corr_{window}d.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

def plot_tweet_volume(df):
    vol = df[["date", "n_tweets"]].drop_duplicates(subset=["date"]).copy()
    vol = vol.set_index("date").sort_index()
    plt.figure(figsize=(12, 4))
    plt.plot(vol.index, vol["n_tweets"], color="slategray", linewidth=1.4)
    plt.title("Total Tweet Volume per Day")
    plt.xlabel("Date")
    plt.ylabel("Tweets")
    plt.tight_layout()
    out = FIG_DIR / "tweet_volume_timeseries.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)

def main():
    df = load_all()
    plot_correlation_bars()
    plot_sentiment_over_all_returns(df)
    plot_heatmap_single_sentiment(df)

    for sec in ["Technology", "Energy", "Financials", "All_Sectors", "Gold_Index_Fund", "United_States_Oil_Fund"]:
        plot_sector_dual_axis(df, sec)
        plot_sector_lag_scatter(df, sec)
        plot_rolling_correlation(df, window=30, sector_name=sec)

    plot_tweet_volume(df)

if __name__ == "__main__":
    main()