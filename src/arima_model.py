import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Load the cleaned dataset
data = pd.read_csv("data/aapl.csv", skiprows=2, index_col=0, parse_dates=True)
data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

# Use only the 'Close' price
close = data['Close']

# Step 1: Check for stationarity (ADF Test)
result = adfuller(close)
print("ADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] < 0.05:
    print("✅ Data is stationary — proceed with ARIMA")
else:
    print("⚠️ Data is non-stationary — differencing will be needed")

# Step 2: Apply ARIMA model (p=5, d=1, q=0 as an example)
model = ARIMA(close, order=(5, 1, 0))
fitted = model.fit()

# Step 3: Forecast next 30 days
forecast = fitted.forecast(steps=30)
print("\nNext 30-Day Forecast:")
print(forecast)

# Step 4: Plot actual vs forecast
plt.figure(figsize=(14, 6))
plt.plot(close, label='Actual')
plt.plot(forecast.index, forecast, label='Forecast (Next 30 days)', color='red')
plt.title("ARIMA Forecast vs Actual Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
