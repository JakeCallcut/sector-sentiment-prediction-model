from yfinance_utils import getDataForDateRange

debug = True
save_to_file = True
file_path = "../../data/yfinance/"
start_date = "2024-01-01"
end_date = "2024-02-01"

# Map of S&P 500 sectors to their Select Sector SPDR ETF tickers
sector_to_ticker = {
    "All": "SPY",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}

#Reverse map for ETF ticker lookup
ticker_to_sector = {v: k for k, v in sector_to_ticker.items()}

def main():

    for key, ticker in sector_to_ticker.items():
        data = getDataForDateRange(ticker, start_date, end_date)

        if debug:
            print(data)
        
        if save_to_file:
            filename = f"{ticker}_{start_date}_to_{end_date}.csv"
            data.to_csv(f"{file_path}{filename}", index=False)
            print(f"Saved {key} ({ticker}) data to {file_path}{filename}")

if __name__ == "__main__":
    main()