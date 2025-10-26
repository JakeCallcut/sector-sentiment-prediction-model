import yfinance as yf

def getDataForDateRange(ticker: str, start: str, end: str):
    data = yf.download(ticker, start, end, auto_adjust=False, progress=True)
    return data


