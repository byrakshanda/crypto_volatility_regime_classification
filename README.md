# Cryptocurrency Volatility Regime Classification (KNN)

## What this project does
Classifies Bitcoin trading days into "Low Volatility" or "High Volatility" regimes 
using K-Nearest Neighbors, and predicts the next day's volatility regime based on 
today's market data.

## Dataset
Bitcoin minute-level price data for 2017 (BTC-2017 per min.csv), resampled into 
365 daily observations (open, high, low, close, volume).

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn, Jupyter Notebook

## What I did
- Converted minute-level Bitcoin data into daily OHLCV data using resampling
- Engineered a `volatility` feature: (high - low) / close
- Labeled each day as Low or High volatility using the median volatility as the 
  cutoff (created a balanced target)
- Shifted the target by one day so the model predicts **tomorrow's** volatility 
  regime from today's data (avoiding data leakage)
- Engineered a `daily_return` feature: (close - open) / open
- Used a time-based train/test split (no shuffling, since this is time-series data)
- Tested KNN with K values from 1 to 10 and compared Accuracy, Precision, Recall, 
  F1, and ROC-AUC for each
- Used an error-rate vs K plot to pick the best K

## Results
| K Value | Accuracy | Precision | Recall | F1 Score |
|---------|----------|-----------|--------|----------|
| 5       | 0.712    | ~0.82     | ~0.77  | ~0.79    |
| 7       | 0.712    | ~0.80     | ~0.79  | ~0.79    |

**Best K:** 5 or 7 (both gave the lowest error rate and best balance of metrics)

## Key Insight
Trading volume had the strongest correlation with volatility (0.87) among all 
market variables. The model shows time-based patterns in volatility can be 
predicted reasonably well one day ahead using recent price/volume behavior.

## How to run it
Open `crypto_volatility_regime_classification.ipynb` in Jupyter Notebook or 
Google Colab and run all cells.                                                                                                         
## Dataset
This project uses minute-level Bitcoin (BTC-USD) price data for 2017 (~525,000 rows). 
The raw file is too large to include in this repository. Similar historical Bitcoin 
data can be found on Kaggle or CryptoDataDownload — place the CSV file 
(`BTC-2017 per min.csv`) in the same folder as the notebook before running it.
