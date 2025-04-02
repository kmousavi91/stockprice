import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Download Stock Data
symbol = "AAPL"  # Change to any stock symbol
data = yf.download(symbol, start="2015-01-01", end="2023-01-01")

# Step 2: Preprocess Data
data = data[['Close']]
data.dropna(inplace=True)
data['Returns'] = data['Close'].pct_change()
data.dropna(inplace=True)

# Step 3: Train-Test Split
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Step 4: Train ARIMA Model
order = (5,1,0)  # ARIMA(p,d,q) parameters
model = ARIMA(train['Close'], order=order)
model_fit = model.fit()

# Step 5: Make Predictions
forecast = model_fit.forecast(steps=len(test))

# Step 6: Evaluate Model
mae = mean_absolute_error(test['Close'], forecast)
mse = mean_squared_error(test['Close'], forecast)
rmse = np.sqrt(mse)
print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}')

# Step 7: Visualize Predictions
plt.figure(figsize=(10, 5))
plt.plot(train.index, train['Close'], label='Training Data', color='blue')
plt.plot(test.index, test['Close'], label='Actual Prices', color='green')
plt.plot(test.index, forecast, label='Predicted Prices', color='red', linestyle='dashed')
plt.legend()
plt.title(f'{symbol} Stock Price Forecast using ARIMA')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()
