import numpy as np
import pandas as pd

def compute_log_returns(prices):
    return np.log(prices / prices.shift(1))

def compute_future_returns(prices, horizon=5):
    return prices.pct_change(horizon).shift(-horizon)

def volatility_features(returns):
    vol_5d = returns.rolling(5).std().add_suffix("_vol_5d")
    vol_21d = returns.rolling(21).std().add_suffix("_vol_21d")
    ewm_vol_21d = returns.ewm(span=21).std().add_suffix("_ewm_vol_21d")
    return pd.concat([vol_5d, vol_21d, ewm_vol_21d], axis=1)

def momentum_features(prices):
    mom_5d = (prices / prices.shift(5) - 1).add_suffix("_momentum_5d")
    mom_21d = (prices / prices.shift(21) - 1).add_suffix("_momentum_21d")
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    macd = (ema12 - ema26).add_suffix("_macd")
    macd_signal = macd.ewm(span=9).mean().add_suffix("_macd_signal")
    return pd.concat([mom_5d, mom_21d, macd, macd_signal], axis=1)

def trend_features(prices):
    sma_10 = prices.rolling(10).mean().add_suffix("_sma_10")
    sma_50 = prices.rolling(50).mean().add_suffix("_sma_50")
    ema_10 = prices.ewm(span=10).mean().add_suffix("_ema_10")
    ema_50 = prices.ewm(span=50).mean().add_suffix("_ema_50")
    price_sma10_ratio = (prices / prices.rolling(10).mean()).add_suffix("_price_sma10_ratio")
    sma10_sma50_ratio = (prices.rolling(10).mean() / prices.rolling(50).mean()).add_suffix("_sma10_sma50_ratio")
    return pd.concat([sma_10, sma_50, ema_10, ema_50, price_sma10_ratio, sma10_sma50_ratio], axis=1)

def volume_features(volume):
    vol_avg_10d = volume.rolling(10).mean().add_suffix("_vol_avg_10d")
    vol_ratio_10d = (volume / volume.rolling(10).mean()).add_suffix("_vol_ratio_10d")
    return pd.concat([vol_avg_10d, vol_ratio_10d], axis=1)

def build_features(prices, volume=None, esg_scores=None):
    """
    Returns a tidy DataFrame with features in logical order:
    Date | Ticker | Price | Volume | Trend | Momentum | Volatility | ESG_score
    """
    returns = compute_log_returns(prices)
    vol_feat = volatility_features(returns)
    mom_feat = momentum_features(prices)
    trend_feat = trend_features(prices)
    volu_feat = volume_features(volume) if volume is not None else None

    prices_renamed = prices.add_suffix("_price")
    if volume is not None:
        volume_renamed = volume.add_suffix("_volume")

    to_concat = [prices_renamed]
    if volume is not None:
        to_concat.append(volume_renamed)
    to_concat += [trend_feat, mom_feat, vol_feat]
    feature_df = pd.concat(to_concat, axis=1)
    feature_df = feature_df.dropna(how="any").reset_index()  # Date as column

    id_vars = ['Date']
    value_vars = [col for col in feature_df.columns if col != 'Date']
    tidy_df = feature_df.melt(id_vars=id_vars, value_vars=value_vars, var_name="Ticker_Feature", value_name="Value")
    tidy_df[['Ticker', 'Feature']] = tidy_df['Ticker_Feature'].str.extract(r'(.+?)_(.+)')
    tidy_df = tidy_df.drop(columns='Ticker_Feature')

    final_df = tidy_df.pivot_table(index=['Date', 'Ticker'], columns='Feature', values='Value').reset_index()

    if esg_scores is not None:
        final_df = final_df.merge(esg_scores, on='Ticker', how='left')

    cols_order = ['Date', 'Ticker']
    # Price and volume
    if 'price' in final_df.columns:
        cols_order.append('price')
    if volume is not None and 'volume' in final_df.columns:
        cols_order.append('volume')
    # Trend features
    trend_cols = ['ema_10', 'ema_50', 'sma_10', 'sma_50', 'price_sma10_ratio', 'sma10_sma50_ratio']
    cols_order += [c for c in trend_cols if c in final_df.columns]
    # Momentum features
    mom_cols = ['momentum_5d', 'momentum_21d', 'macd', 'macd_macd_signal']
    cols_order += [c for c in mom_cols if c in final_df.columns]
    # Volatility features
    vol_cols = ['vol_5d', 'vol_21d', 'ewm_vol_21d']
    cols_order += [c for c in vol_cols if c in final_df.columns]
    # ESG last
    if esg_scores is not None and 'esg_score' in final_df.columns:
        cols_order.append('esg_score')

    final_df = final_df[cols_order]
    return final_df