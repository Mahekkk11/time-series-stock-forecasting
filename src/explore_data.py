import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV with correct index and skip junk rows
data = pd.read_csv("data/aapl.csv", skiprows=2, index_col=0, parse_dates=True)

# Rename columns
data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']

# Basic info
print("Data Info:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

# Display first few rows
print("\nFirst 5 rows:")
print(data.head())
# Plot closing price
data['Close'].plot(figsize=(14, 6), title='AAPL Closing Price Over Time')
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.show()
