import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
nltk.download('stopwords')
from nltk.corpus import stopwords
import pyLDAvis 
#import pyLDAvis.sklearn 
print(pd.__version__)
#loading dataset
#file_path=r'C:\\Users\\Lalaa\\Downloads\\Data-20241211T111619Z-001\\Data\\raw_analyst_ratings.csv'
df=pd.read_csv('C:\\Users\\Lalaa\\Documents\\GitHub\\raw_analyst_ratings.csv')
print('print data')
#calculating length of headlines 
if 'headline' in df.columns:
    df['headline_length'] = df['headline'].apply(len)
    headline_length_stats = df['headline_length'].describe()
    print("Headline Length Statistics:\n", headline_length_stats)
else:
    print("No 'headline' column found.")
#calculating count of article in publisher columns 
if 'publisher' in df.columns:
    articles_per_publisher = df['publisher'].value_counts()
    print("\nArticles per Publisher:\n", articles_per_publisher)
else:
    print("No 'publisher' column found.")

#processessing publication date
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M', errors='coerce')
# Check for any conversion issues
if df['date'].isnull().any():
    print("Some dates could not be converted:\n", df[df['date'].isnull()])
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])  # Convert to datetime
    df['publication_day'] = df['date'].dt.date
    df['publication_month'] = df['date'].dt.to_period('M')

    # Count articles per publication day
    articles_per_day = df['publication_day'].value_counts().sort_index()
    print("\nArticles per Day:\n", articles_per_day)

    # Count articles per month
    articles_per_month = df['publication_month'].value_counts().sort_index()
    print("\nArticles per Month:\n", articles_per_month)
else:
    print("No 'date' column found.")

#sentiment aanalysis using NLTK 
nltk.download('vader_lexicon')
sa=SentimentIntensityAnalyzer()

#let create sentiment analyzer function 
def analyze_sentiment(text):
    score=sa.polarity_scores(text)
    if score['compound']>0:
        sentiment='Positive'
    elif score['compound']<0:
        sentiment='Negetive'
    else:
        sentiment='Neutral'
    return score['compound'], sentiment

df['sentiment_score'], df['sentiment'] = zip(*df['headline'].apply(analyze_sentiment))
print(df[['headline', 'sentiment_score', 'sentiment']].head())

sentiment_distribution = df['sentiment'].value_counts()
print(sentiment_distribution)
sns.countplot(data=df, x='sentiment')
plt.title('Sentiment Distribution of Headlines')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

#topic modeling 
stop_words=stopwords.words('english')
vectorizer=CountVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(df['headline'])

# Fit LDA model
lda = LatentDirichletAllocation(n_components=5, random_state=42)  
lda.fit(X)

# Display topics
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

# Display the top words for each topic
no_top_words = 5
display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)
# Visualize the topics
panel = pyLDAvis.sklearn.prepare(lda, X, vectorizer, mds='tsne')
pyLDAvis.show(panel)
# Visualize the topics
panel = pyLDAvis.sklearn.prepare(lda, X, vectorizer, mds='tsne')
pyLDAvis.show(panel)
topic_distribution = lda.transform(X)
# Create a DataFrame for better visualization
distribution_df = pd.DataFrame(topic_distribution, columns=[f'Topic {i+1}' for i in range(lda.n_components)])
print(distribution_df.head())
# Create a bar plot for topic distribution
#Calculate the average topic distribution
average_distribution = distribution_df.mean()

# Plot the average topic distribution
plt.figure(figsize=(10, 6))
average_distribution.plot(kind='bar')
plt.title('Average Topic Distribution Across Documents')
plt.xlabel('Topics')
plt.ylabel('Average Contribution')
plt.xticks(rotation=0)
plt.show()

#Time series analysis 
# Convert 'date' to datetime format
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %H:%M', errors='coerce') 
# Set the date as the index
df.set_index('date', inplace=True)

# Resample to count publications, e.g., daily frequency
publication_counts = df.resample('D').size()  # Change 'D' to 'W' for weekly, 'M' for monthly
# Plot the publication frequency
plt.figure(figsize=(12, 6))
plt.plot(publication_counts.index, publication_counts.values, marker='o', linestyle='-')
plt.title('Publication Frequency Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Publications')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

#identfy spikes and market event 
# Example market events (you would replace these with real events)
market_events = {
    pd.Timestamp('2020-05-01'): 'Market Rally',
    pd.Timestamp('2020-06-15'): 'Earnings Reports',
    # if there is more event we will add here
}

# Plot with annotations
plt.figure(figsize=(12, 6))
plt.plot(publication_counts.index, publication_counts.values, marker='o', linestyle='-')

# Annotate market events
for event_date, event_name in market_events.items():
    plt.annotate(event_name, xy=(event_date, publication_counts[event_date]), 
                 xytext=(event_date, publication_counts[event_date] + 5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'))

plt.title('Publication Frequency Over Time with Market Events')
plt.xlabel('Date')
plt.ylabel('Number of Publications')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# Extract the hour from the date
df['hour'] = df.index.hour

# Count publications by hour
hourly_counts = df['hour'].value_counts().sort_index()

# Plot publishing times
plt.figure(figsize=(12, 6))
hourly_counts.plot(kind='bar')
plt.title('Publication Frequency by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Publications')
plt.xticks(rotation=0)
plt.grid()
plt.tight_layout()
plt.show()

# Count the number of articles by publisher
publisher_counts = df['publisher'].value_counts()

# Display the top publishers
print(publisher_counts.head(10))

# Define categories and keywords
categories = {
    'Earnings': ['earnings', 'profit', 'loss', 'report'],
    'Market Trends': ['market', 'stocks', 'indices'],
    'Mergers & Acquisitions': ['merger', 'acquisition', 'buyout'],
    'Regulatory': ['FDA', 'regulation', 'policy'],
}

# Function to categorize headlines
def categorize_headline(headline):
    for category, keywords in categories.items():
        if any(keyword.lower() in headline.lower() for keyword in keywords):
            return category
    return 'Other'

# Apply categorization
df['category'] = df['headline'].apply(categorize_headline)

# Count categories by publisher
category_counts = df.groupby(['publisher', 'category']).size().unstack(fill_value=0)

# Display the category counts
print(category_counts)

# Function to extract domain from email
def extract_domain(email):
    return email.split('@')[-1] if '@' in email else email

# Apply the domain extraction
df['domain'] = df['publisher'].apply(extract_domain)

# Count the number of articles by domain
domain_counts = df['domain'].value_counts()

# Display the top domains
print(domain_counts.head(10))

# Plot top publishers
plt.figure(figsize=(12, 6))
publisher_counts.head(10).plot(kind='bar')
plt.title('Top Publishers by Article Count')
plt.xlabel('Publisher')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Plot category counts by publisher
category_counts.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Article Category Counts by Publisher')
plt.xlabel('Publisher')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
