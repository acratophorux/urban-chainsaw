import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz
import joblib
import os

def load_preprocessed_data(filepath):
    """Load preprocessed data from csv file"""
    df = pd.read_csv(filepath, low_memory=False)
    df['cleaned_review'] = df['cleaned_review'].fillna('')
    return df


def create_tfidf_features(df, text_column, max_features=5000):
    """Create TF-IDF features from a column"""
    tfidf = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = tfidf.fit_transform(df[text_column])

    feature_names = tfidf.get_feature_names_out()

    return tfidf_matrix, tfidf, feature_names

def save_features(tfidf_matrix, tfidf_vectorizer, feature_names, matrix_file, vectorizer_file, names_file):
    """Save TF-IDF matrix, feature names and vectorizer"""
    save_npz(matrix_file, tfidf_matrix)

    joblib.dump(tfidf_vectorizer, vectorizer_file)

    np.save(names_file, feature_names)


def main():
    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    df = load_preprocessed_data(os.path.join(ROOT_DIR, 'data/processed/processed_luxury_beauty_5.csv'))

    # create tf-idf features
    tfidf_matrix, tfidf_vectorizer, feature_names = create_tfidf_features(df, 'cleaned_review')

    # save the features
    MATRIX_FILEPATH = os.path.join(ROOT_DIR, 'data/tfidf/tfidf_matrix.npz')
    VECTORIZER_FILEPATH = os.path.join(ROOT_DIR, 'data/tfidf/tfidf_vectorizer.joblib')
    FEATURE_NAMES_FILEPATH = os.path.join(ROOT_DIR, 'data/tfidf/feature_names.npy')
    save_features(tfidf_matrix, tfidf_vectorizer, feature_names, 
                  MATRIX_FILEPATH, 
                  VECTORIZER_FILEPATH,
                  FEATURE_NAMES_FILEPATH)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Top 10 features:")
    print(feature_names[:10])

if __name__ == "__main__":
    main()