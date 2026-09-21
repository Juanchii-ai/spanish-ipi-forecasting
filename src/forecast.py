from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    mask = actual.ne(0)
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": mean_squared_error(actual, predicted) ** 0.5,
        "mape": float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--date-column", required=True)
    parser.add_argument("--value-column", required=True)
    parser.add_argument("--horizon", type=int, default=12)
    args = parser.parse_args()

    frame = pd.read_csv(args.data)
    frame[args.date_column] = pd.to_datetime(frame[args.date_column])
    series = frame.set_index(args.date_column)[args.value_column].astype(float).asfreq("MS").interpolate()
    train, test = series.iloc[:-args.horizon], series.iloc[-args.horizon:]
    adf = adfuller(train.dropna())
    print(f"ADF statistic={adf[0]:.4f}; p-value={adf[1]:.4f}")

    hw = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
    sarimax = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    forecasts = {
        "holt_winters": hw.forecast(args.horizon),
        "sarimax": sarimax.forecast(args.horizon),
    }
    for name, forecast in forecasts.items():
        print(name, metrics(test, forecast))
    print("SARIMAX Ljung-Box")
    print(acorr_ljungbox(sarimax.resid.dropna(), lags=[12], return_df=True))


if __name__ == "__main__":
    main()

