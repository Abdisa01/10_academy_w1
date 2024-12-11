import pandas as pd
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

