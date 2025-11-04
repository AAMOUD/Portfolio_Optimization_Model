import yfinance as yf
import pandas as pd
import numpy as np
import os

def get_price_volume(tickers, start="2001-01-01", end="2025-01-01", interval="1d"):
    """
    Download adjusted close prices for a list of tickers robustly.

    Parameters:
    - tickers: list of tickers
    - start, end: date strings
    - interval: string (1d, 1wk, 1mo)

    Returns:
    - prices: DataFrame of adjusted close prices
    - volumes: DataFrame of volumes
    """
    data = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.levels[0]:
            prices = data['Adj Close']
        else:
            prices = data['Close']
    else:
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']

    prices = prices.dropna(how='all').fillna(method='ffill')
    volumes = data['Volume']

    return prices, volumes

def save_ohlc_data(tickers, save_path, start="2001-01-01", end="2025-01-01", interval="1d"):
    """
    Download and save OHLCV data for each ticker, with clean single-level headers.

    Parameters:
    - tickers: list of tickers
    - save_path: directory to save CSV files
    - start, end: date range
    - interval: data interval ('1d', '1wk', '1mo')
    """
    os.makedirs(save_path, exist_ok=True)

    for ticker in tickers:
        try:
            print(f"Downloading {ticker} ...")
            data = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False)

            if data.empty:
                print(f"No data found for {ticker}")
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data.reset_index()

            cols_to_keep = ["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]
            data = data[cols_to_keep]

            file_path = os.path.join(save_path, f"{ticker}.csv")
            data.to_csv(file_path, index=False)

            print(f"Saved clean data for {ticker} → {file_path}")

        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

    print("\nAll downloads complete!")


def get_sample_esg_scores(tickers):
    """
    Return a dummy ESG score for demonstration purposes.
    """
    np.random.seed(42)
    return pd.DataFrame({
        "ticker": tickers,
        "esg_score": np.random.uniform(0.4, 0.9, len(tickers))
    })
