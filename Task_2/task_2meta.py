import os
import pandas as pd
import pkg_resources
import talib
import matplotlib.pyplot as plt
from pynance import data

# Load historical data from CSV
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/GitHub/weak_1_data/yfinance_data/yfinance_data/META_historical_data.csv')

# Display the first few rows of the DataFrame
print(df.head())

# Ensure the date column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Calculate Moving Averages
df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)

# Calculate RSI
df['RSI'] = talib.RSI(df['Close'], timeperiod=14)

# Calculate MACD
df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

# Display the DataFrame with indicators
print(df[['Close', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'MACD_signal']].tail())

# Attempt to retrieve financial data
try:
    stock_data = data.get("AAPL")  # This should be a method to get stock data
    print("Stock Data:\n", stock_data)  # Display fetched stock data
except Exception as e:
    print(f"Error retrieving stock data: {e}")

# Plot Closing Price and Moving Averages
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Close Price')
plt.plot(df['SMA_50'], label='50-Day SMA', linestyle='--')
plt.plot(df['SMA_200'], label='200-Day SMA', linestyle='--')
plt.title('Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# Plot RSI
plt.figure(figsize=(14, 5))
plt.plot(df['RSI'], label='RSI', color='blue')
plt.axhline(70, linestyle='--', alpha=0.5, color='red')
plt.axhline(30, linestyle='--', alpha=0.5, color='green')
plt.title('Relative Strength Index (RSI)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.grid()
plt.show()