import os
import pandas as pd
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# download nltk punkt and stopwords data
nltk.download('punkt')
nltk.download('stopwords')

def load_data_from_json(filepath):
    """Load dataset from given JSON file"""
    df =  pd.read_json(filepath, lines=True)
    cols_to_drop = ['verified', 'reviewTime', 'reviewerID', 'asin', 'style', 
                'reviewerName', 'unixReviewTime', 'image', 'vote']
    return df.drop(columns=cols_to_drop, axis=1)
def clean_text(text):
    """Clean text data: remove html tags, special chars, and convert to lowercase"""
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

def remove_stopwords_from_tokens(tokens):
    """Remove stops words from tokens"""
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
   tokens = remove_stopwords_from_tokens(tokens)
   tokens = stem_tokens(tokens)
   return ' '.join(tokens)

def preprocess_data(df):
    """Dataframe preprocessing pipeline"""

    df['cleaned_review'] = df['reviewText'].fillna('').apply(preprocess_text).fillna('')
    df['cleaned_summary'] = df['summary'].fillna('').apply(preprocess_text).fillna('')
    df['cleaned_review_length'] = df['cleaned_review'].str.len()
    return df

def explore_data(df):
    """Perform data exploration of preprocessed data"""
    print("Dataset shape:", df.shape)
    print("\nColumn names:", df.columns.tolist())
    print("\nSample preprocessed reviews:")
    print(df['cleaned_review'].head())
    print("\nDistribution of ratings:")
    print(df['overall'].value_counts().sort_index())
    print("\nReview length statistics:")
    print(df['cleaned_review_length'].describe())

def main():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    # print(root_dir)
    print("Loading data from json file...")
    df = load_data_from_json(os.path.join(root_dir, 'data/Luxury_Beauty_5.json'))
    # explore_data(df)


    print("Processing data...")
    df = preprocess_data(df)
    # explore_data(df)

    print("Saving to csv file...")
    df.to_csv(os.path.join(root_dir, 'data/processed/processed_luxury_beauty_5.csv'))
    print("Done.")

if __name__ == "__main__":
    main()
