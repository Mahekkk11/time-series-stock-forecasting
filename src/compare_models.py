import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load real closing price
data = pd.read_csv("data/aapl.csv", skiprows=2, index_col=0, parse_dates=True)
data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
close = data['Close'].reset_index(drop=True)

# Simulated predictions (replace with real ones later)
# For now, let's use last 100 points as comparison range
true_values = close[-100:].values

# You should replace these with real prediction results from models
# For demo purposes, we'll create dummy predictions
arima_pred = true_values + np.random.normal(0, 2, size=len(true_values))
prophet_pred = true_values + np.random.normal(0, 3, size=len(true_values))
lstm_pred = true_values + np.random.normal(0, 4, size=len(true_values))

# Evaluation function
def evaluate(true, pred, model_name):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    return mae, rmse

# Evaluate each model
evaluate(true_values, arima_pred, "ARIMA")
evaluate(true_values, prophet_pred, "Prophet")
evaluate(true_values, lstm_pred, "LSTM")

# Plot comparison
plt.figure(figsize=(14, 6))
plt.plot(true_values, label='Actual', linewidth=2)
plt.plot(arima_pred, label='ARIMA')
plt.plot(prophet_pred, label='Prophet')
plt.plot(lstm_pred, label='LSTM')
plt.title("Model Prediction Comparison")
plt.xlabel("Time Steps")
plt.ylabel("Closing Price")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
