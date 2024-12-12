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
print(pd.__version__)
#loading dataset
#file_path=r'C:\\Users\\Lalaa\\Downloads\\Data-20241211T111619Z-001\\Data\\raw_analyst_ratings.csv'
df=pd.read_csv('C:\\Users\\Abdilala\\Documents\\GitHub\\10_academy_w1\\raw_analyst_ratings.csv')
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
#nltk.download('vader_lexicon')
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