import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

def prepare_ml_data(df, target_horizon=1):
    """Prepare X, y for ML models (predicting next-day returns)."""
    df = df.copy()
    df["target_return"] = df.groupby("Ticker")["price"].shift(-target_horizon) / df["price"] - 1
    df = df.dropna()
    
    feature_cols = [
        "price", "volume", "ema_10", "ema_50", "sma_10", "sma_50",
        "price_sma10_ratio", "sma10_sma50_ratio",
        "momentum_5d", "momentum_21d",
        "macd", "macd_macd_signal",
        "vol_5d", "vol_21d", "ewm_vol_21d",
        "esg_score"
    ]
    
    X = df[feature_cols]
    y = df["target_return"]
    return X, y, df

def train_models(X, y, test_size=0.2, random_state=42):
    """Train linear regression, random forest, and MLP."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=random_state),
        "MLP": MLPRegressor(hidden_layer_sizes=(64,32), max_iter=300, random_state=random_state)
    }
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        results[name] = {"model": model, "mse": mse, "r2": r2}
    
    return results, scaler
