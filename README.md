# Spanish Industrial Production Forecasting

Time-series analysis of Spain's Industrial Production Index (IPI), comparing exponential-smoothing and SARIMAX approaches.

## Workflow

1. Validate and regularise the monthly series.
2. Explore trend and seasonality.
3. Test stationarity with the Augmented Dickey–Fuller test.
4. Fit Holt–Winters and SARIMAX models.
5. Compare MAE, RMSE and MAPE on a holdout period.
6. Diagnose residual autocorrelation with the Ljung–Box test.

## Run

```bash
python src/forecast.py --data data/ipi.csv --date-column date --value-column ipi
```

## Technologies

Python · Pandas · Statsmodels · Time-series diagnostics · Forecasting

