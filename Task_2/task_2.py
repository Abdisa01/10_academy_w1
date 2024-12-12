import pandas as pd 
import talib
from pynance import Stock
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\Lalaa\\Documents\\GitHub\\AAPL_historical_data.csv')

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

# Example: Get financial metrics for a specific stock
stock = Stock('AAPL')  # Replace with your stock symbol
financials = stock.financials
print(financials)

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
plt.plot(df['RSI'], label='RSI', color='orange')
plt.axhline(70, linestyle='--', alpha=0.5, color='red')
plt.axhline(30, linestyle='--', alpha=0.5, color='green')
plt.title('Relative Strength Index (RSI)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.grid()
plt.show()