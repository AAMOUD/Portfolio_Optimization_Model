# Machine Learning–Driven Portfolio Optimization with ESG Focus

## Phase 1: Data Collection & Feature Engineering

This phase focuses on acquiring financial data and creating a feature-rich dataset for downstream machine learning models and portfolio optimization. The output of this phase is a clean, processed CSV containing price, technical indicators, and ESG scores.

---

## Dataset Description

The final CSV file for this phase is:

**`features_final.csv`**

### Columns

| Column | Description |
|--------|-------------|
| Date | Trading date (YYYY-MM-DD) |
| Ticker | Stock ticker symbol |
| price | Adjusted closing price |
| volume | Daily trading volume |
| ema_10 | 10-day exponential moving average |
| ema_50 | 50-day exponential moving average |
| sma_10 | 10-day simple moving average |
| sma_50 | 50-day simple moving average |
| price_sma10_ratio | Ratio of price to 10-day SMA |
| sma10_sma50_ratio | Ratio of 10-day SMA to 50-day SMA |
| momentum_5d | 5-day price momentum |
| momentum_21d | 21-day price momentum |
| macd | Moving Average Convergence Divergence |
| macd_macd_signal | MACD signal difference |
| vol_5d | 5-day rolling volatility |
| vol_21d | 21-day rolling volatility |
| ewm_vol_21d | 21-day exponentially weighted volatility |
| esg_score | ESG score (0–1) representing environmental, social, and governance performance |

---

## Data Sources

- **Price and Volume Data:** Yahoo Finance via `yfinance` API.
- **ESG Score:** Randomly generated proxy values for demonstration. In a full-scale project, replace with actual ESG data from providers such as MSCI, Sustainalytics, or Refinitiv.

---

## Feature Engineering

- **Returns & Volatility:** Computed percentage changes and rolling volatilities.
- **Technical Indicators:** EMAs, SMAs, momentum, MACD.
- **Ratios:** Price/SMA and SMA ratios to capture trends.
- **ESG Scores:** Incorporated as a static feature to enable ESG-aware optimization in future phases.

---

## How to Use

1. Clone this repository and navigate to the project folder.
2. Install dependencies listed in `requirements.txt`.
3. Load the CSV in Python:

```python
import pandas as pd
df = pd.read_csv("data/processed/features_final.csv")
print(df.head())
```
# Phase 2: Machine Learning Prediction of Returns

This phase focuses on training machine learning models to predict next-day stock returns using the features engineered in **Phase 1**. The predicted returns will later feed into portfolio optimization with ESG considerations.

---

## Objectives

- Build predictive models for stock returns using historical price, volume, technical indicators, and ESG scores.
- Evaluate model performance and select the best-performing model.
- Save predicted returns for use in portfolio optimization.
- Visualize model performance using interactive plots.

---

## Dataset

**Input:** `features_final.csv` (from Phase 1), containing:

**Output:** `predicted_returns.csv`, which adds:

- `predicted_return`: ML model prediction of next-day return.

---

## Models Used

- **Linear Regression** — baseline model
- **Random Forest Regressor** — non-linear tree-based model
- **MLP Neural Network** — optional deep learning model

---

## Steps

1. **Prepare ML data**  
   - Create `target_return` = next-day return.  
   - Select features and target for training.

2. **Train & Evaluate Models**  
   - Split data into training and test sets.  
   - Standardize features with `StandardScaler`.  
   - Train Linear Regression, Random Forest, and MLP.  
   - Evaluate using Mean Squared Error (MSE) and R² score.  
   - Select the model with lowest MSE as `best_model`.

3. **Save Predictions**  
   - Add `predicted_return` column to DataFrame.  
   - Save full dataset with predictions:  
     ```bash
     data/processed/predicted_returns.csv
     ```

4. **Visualize Results**  
   - **Scatter Plot:** Actual vs predicted returns to see overall accuracy.  
   - **Line Plot:** Actual vs predicted returns over time for one or multiple tickers.  
   - **Interactive Plot (Plotly):** Zoom, hover, and filter by ticker.  
   - Optionally, save interactive plots as HTML for GitHub display.

---

## Key Insights

- Random Forest or MLP often outperform linear regression for predicting returns due to non-linear relationships in the data.  
- ESG scores can be included as static features to make the predictions ESG-aware.  
- Interactive plots help visualize model performance over time for each stock.

---

## Next Steps

- Feed the predicted returns into **Phase 3: Portfolio Optimization**.  
- Apply ESG constraints to optimize asset allocation based on ML predictions.
