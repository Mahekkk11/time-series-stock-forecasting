import yfinance as yf
import os

# Create the 'data' folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Download stock data
data = yf.download("AAPL", start="2020-01-01", end="2024-12-31")

# Save it to CSV
data.to_csv("data/aapl.csv")

print("✅ Data downloaded and saved to data/aapl.csv")
