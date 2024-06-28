import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download nltk 'punkt' and 'stopwords' data
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    """Clean the text data"""

    # remove html tags
    text = re.sub(r'<.*?>', '', text)
    
    # remove any special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # convert to lowercase
    text = text.lower()
    
    return text

def tokenize_text(text):
    """Tokenize the text data"""
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Remove stop words"""
    # stop words carry very little to no information
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def stem_tokens(tokens):
    """Stem the tokens"""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def preprocess_text(text):
    """Text preprocessing pipeline"""
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return ' '.join(tokens)

def preprocess_data(df):
    """Dataframe preprocessing pipeline"""

    # preprocess the review text
    print('cleaning review text...')
    df['cleaned_review'] = df['reviewText'].fillna('').apply(preprocess_text)
    
    # preprocess the summary text
    print('cleaning summary text...')
    df['cleaned_summary'] = df['summary'].fillna('').apply(preprocess_text)

    # create review_length feature
    print('adding review_length feature...')
    df['review_length'] = df['reviewText'].fillna('').str.len()

    # extract year from time data
    print('adding review_year feature...')
    df['review_year'] = pd.to_datetime(df['unixReviewTime'], unit='s').dt.year

    # convert verified to integer
    print('converting verified to int...')
    df['verified_purchase'] = df['verified'].astype(int)

    print('Done.')
    return df

def load_data(filepath):
    """Load Dataset from a JSON file"""
    return pd.read_json(filepath, lines=True)

def explore_data(df):
    """Perform data exploration of preprocessed data"""
    print("Dataset shape:", df.shape)
    print("\nColumn names:", df.columns.tolist())
    print("\nSample preprocessed reviews:")
    print(df['cleaned_review'].head())
    print("\nDistribution of ratings:")
    print(df['overall'].value_counts().sort_index())
    print("\nReview length statistics:")
    print(df['review_length'].describe())

def main():
    print("Loading data...")
    df = load_data("../data/Office_Products_5.json")
    print("Data Loaded.")

    print("Original reviews:")
    print(df['reviewText'].head())

    print("Preprocessing the data...")
    df = preprocess_data(df)
    print("Done.")

    print("\nCleaned reviews:")
    print(df['cleaned_review'].head())
    
    explore_data(df)

    print('Saving to csv...')
    df.to_csv('../data/preprocessed_office_products.csv', index=False)
    print('Done.')
if __name__ == "__main__":
    main()