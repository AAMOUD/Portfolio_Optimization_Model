import yfinance as yf
import pandas as pd
import numpy as np

def get_price_volume(tickers, start="2020-01-01", end="2025-01-01", interval="1d"):
    """
    Download adjusted close prices for a list of tickers robustly.

    Parameters:
    - tickers: list of tickers
    - start, end: date strings
    - interval: string (1d, 1wk, 1mo)

    Returns:
    - DataFrame with tickers as columns and dates as index
    """
    data = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.levels[0]:
            prices = data['Adj Close']
        else:
            prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']

    prices = prices.dropna(how='all')
    prices = prices.fillna(method='ffill')

    volumes = data['Volume']

    return prices,volumes

def get_sample_esg_scores(tickers):
    """
    Return a dummy ESG score for demonstration purposes.
    """
    np.random.seed(42)
    return pd.DataFrame({
        "ticker": tickers,
        "esg_score": np.random.uniform(0.4, 0.9, len(tickers))
    })
