import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
data = pd.read_csv("data/aapl.csv", skiprows=2, index_col=0, parse_dates=True)
data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

# Prophet needs columns: ds (date) and y (value to forecast)
df = data.reset_index()[['Date', 'Close']]
df.columns = ['ds', 'y']  # Prophet expects these column names

# Create and fit the model
model = Prophet(daily_seasonality=True)
model.fit(df)

# Create future dataframe for next 30 days
future = model.make_future_dataframe(periods=30)

# Predict future
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title("Prophet Forecast for AAPL Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.tight_layout()
plt.show()

# Optional: Plot forecast components
model.plot_components(forecast)
plt.tight_layout()
plt.show()
