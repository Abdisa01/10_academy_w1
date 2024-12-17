import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# File paths
news_file = '/content/drive/MyDrive/Colab Notebooks/GitHub/weak_1_data/raw_analyst_ratings.csv/raw_analyst_ratings.csv'
stock_file = '/content/drive/MyDrive/Colab Notebooks/GitHub/weak_1_data/yfinance_data/yfinance_data/TSLA_historical_data.csv'

# Initialize empty DataFrames
news_list = []
stock_list = []

# Load news dataset in chunks
for chunk in pd.read_csv(news_file, chunksize=1000):
    chunk['date'] = pd.to_datetime(chunk['date'], format='%Y-%m-%d %H:%M:%S%z', errors='coerce')
    chunk = chunk[chunk['stock'] == 'A']  # Filter AAPL ticker
    chunk['date'] = chunk['date'].dt.date  # Strip time and timezone to keep only date
    news_list.append(chunk.dropna(subset=['date']))  # Drop rows with NaT in 'date'

# Combine all chunks for the news dataset
news_df = pd.concat(news_list, ignore_index=True)

# Load stock dataset in chunks
for chunk in pd.read_csv(stock_file, chunksize=1000):
    chunk['Date'] = pd.to_datetime(chunk['Date'], format='%m/%d/%Y', errors='coerce')
    chunk = chunk.dropna(subset=['Date'])
    chunk['Date'] = chunk['Date'].dt.date  # Strip time to keep only date
    stock_list.append(chunk)

# Combine all chunks for the stock dataset
stock_df = pd.concat(stock_list, ignore_index=True)

# Rename 'Date' to 'date' for consistency
stock_df.rename(columns={'Date': 'date'}, inplace=True)

# Merge datasets on 'date'
merged_df = pd.merge(news_df, stock_df, on='date', how='inner')

# Display results
print("News DataFrame columns:", news_df.columns)
print("Stock DataFrame columns:", stock_df.columns)
print("Merged DataFrame:\n", merged_df.head())


# Ensure VADER lexicon is downloaded (uncomment if needed)
# nltk.download('vader_lexicon')

# Define the ticker and the time period
ticker = 'A'
start_date = '2019-01-01'
end_date = '2020-12-01'

# Initialize VADER sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Load datasets in chunks to manage memory
news_chunks = []
stock_chunks = []

# Load stock data in chunks
stock_chunk_size = 1000
for stock_chunk in pd.read_csv('/content/drive/MyDrive/Colab Notebooks/GitHub/weak_1_data/yfinance_data/yfinance_data/TSLA_historical_data.csv',
                                chunksize=stock_chunk_size):
    # Convert Date column to datetime (MM/DD/YYYY format)
    stock_chunk['Date'] = pd.to_datetime(stock_chunk['Date'], format='%m/%d/%Y', errors='coerce')
    stock_chunks.append(stock_chunk)

# Concatenate all stock chunks into a single DataFrame
stock_df = pd.concat(stock_chunks, ignore_index=True)

# Filter stock data by date range
stock_df = stock_df[(stock_df['Date'] >= start_date) & (stock_df['Date'] <= end_date)]

# Load news data in chunks
news_chunk_size = 1000
for news_chunk in pd.read_csv('/content/drive/MyDrive/Colab Notebooks/GitHub/weak_1_data/raw_analyst_ratings.csv/raw_analyst_ratings.csv',
                               chunksize=news_chunk_size):
    # Convert date column to datetime (ISO 8601 format)
    news_chunk['date'] = pd.to_datetime(news_chunk['date'], format='"%Y-%m-%d %H:%M:%S%z"', errors='coerce')
    news_chunks.append(news_chunk)

# Concatenate all news chunks into a single DataFrame
news_df = pd.concat(news_chunks, ignore_index=True)

# Check if date conversion was successful
if news_df['date'].isnull().any():
    print("Warning: Some dates in news data are invalid and will be removed.")

# Remove rows with invalid dates
news_df = news_df[news_df['date'].notna()]

# Normalize dates by keeping only the date part
news_df['date'] = news_df['date'].dt.normalize()  # Removes time component
stock_df['Date'] = stock_df['Date'].dt.normalize()  # Removes time component

# Filter news data by date range and ticker
news_df = news_df[(news_df['date'] >= start_date) & (news_df['date'] <= end_date) & (news_df['stock'] == ticker)]

# Merge the datasets on date
merged_df = pd.merge(news_df, stock_df.rename(columns={'Date': 'date'}), on='date', how='inner')

# Perform sentiment analysis
merged_df['Sentiment'] = merged_df['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Aggregate daily sentiments
daily_sentiment = merged_df.groupby('date')['Sentiment'].mean().reset_index()

# Calculate daily returns for stock prices
stock_df['Daily Return'] = stock_df['Close'].pct_change()

# Merge daily sentiment with daily returns
final_df = pd.merge(daily_sentiment, stock_df[['Date', 'Daily Return']], left_on='date', right_on='Date', how='inner')

# Calculate Pearson correlation coefficient
correlation = final_df['Sentiment'].corr(final_df['Daily Return'])

# Display results
print("Aggregated Daily Sentiments:\n", daily_sentiment)
print("Final DataFrame:\n", final_df.head())
print("Pearson Correlation Coefficient between Sentiment and Daily Returns:", correlation)

merged_df['Daily_Return'] = merged_df['Close'].pct_change() 

# Merge daily sentiment scores with daily returns
analysis_df = pd.merge(daily_sentiment, merged_df[['Date', 'Daily_Return']], on='Date', how='inner')
# Calculate correlation
correlation = analysis_df['Sentiment'].corr(analysis_df['Daily_Return'])
print(f"Pearson correlation coefficient: {correlation}")