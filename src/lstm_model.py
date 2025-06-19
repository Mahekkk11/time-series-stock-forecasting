import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess data
data = pd.read_csv("data/aapl.csv", skiprows=2, index_col=0, parse_dates=True)
data.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
close_data = data[['Close']]

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_data)

# Prepare training data (sliding window of 60 timesteps)
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i - time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Reshape input to [samples, time steps, features] for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into train/test
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Predict
y_pred = model.predict(X_test)

# Inverse transform to get real prices
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Plot
plt.figure(figsize=(14,6))
plt.plot(y_test_inv, label="Actual")
plt.plot(y_pred_inv, label="Predicted", color='red')
plt.title("LSTM Model - Actual vs Predicted Closing Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
