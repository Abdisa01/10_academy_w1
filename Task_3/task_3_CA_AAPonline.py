import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob

# Define the ticker and the time period
ticker = 'AAPL'
start_date = '2020-01-01'
end_date = '2023-01-01'

# Fetch the stock data
aapl_data = yf.download(ticker, start=start_date, end=end_date)
aapl_data.head()
aapl_data.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

aapl_data.columns
 dates = aapl_data.index
text_data = pd.DataFrame(dates, columns=['Date'])
 # Generate random headlines
headline_samples = [
    "AAPL hits record high",
    "Concerns over AAPL's future growth",
    "AAPL to unveil new product next month",
    "AAPL reports earnings that exceed forecasts",
    "Market downturn affects AAPL",
    "AAPL invests in renewable energy",
    "New AAPL CEO announced",
    "AAPL faces regulatory scrutiny",
    "AAPL rumored to acquire a tech startup",
    "AAPL's market share grows"
]

# Assign a random headline to each date
text_data['Headline'] = np.random.choice(headline_samples, size=len(text_data))

# Display the first few rows of the updated DataFrame
text_data.head()
text_data.head(30)

def calculate_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Applying the sentiment analysis function to your headlines
text_data['Sentiment'] = text_data['Headline'].apply(calculate_sentiment)

# Aligning sentiment scores with the stock closing prices on the same date
combined_data = text_data.set_index('Date').join(aapl_data['Close'])
combined_data.head()

combined_data['Daily Returns'] = combined_data['Close'].pct_change()
combined_data.head()
correlation = combined_data['Sentiment'].corr(combined_data['Daily Returns'])
print("Correlation between sentiment and daily stock returns:", correlation)
correlation = combined_data['Sentiment'].corr(combined_data['Close'])
print("Correlation between sentiment and closing:", correlation)