from yfinance_utils import getDataForDateRange

save_to_file = False
file_path = "../../data/"

# Map of S&P 500 sectors to their Select Sector SPDR ETF tickers
sector_to_ticker = {
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

    for key in sector_to_ticker:
        print(getDataForDateRange(sector_to_ticker[key], "2024-02-10", "2025-02-03"))

if __name__ == "__main__":
    main()