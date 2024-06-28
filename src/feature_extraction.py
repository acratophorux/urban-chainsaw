import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz
import joblib

def load_preprocessed_data(filepath):
    """Load preprocessed data from csv file"""
    return pd.read_csv(filepath)

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
    df = load_preprocessed_data('../data/preprocessed_office_products.csv')

    # create tf-idf features
    tfidf_matrix, tfidf_vectorizer, feature_names = create_tfidf_features(df, 'cleaned_review')

    # save the features
    save_features(tfidf_matrix, tfidf_vectorizer, feature_names, 
                  '../data/tfidf_matrix.npz', 
                  '../data/tfidf_vectorizer.joblib',
                  '../data/feature_names.npy')
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Top 10 features:")
    print(feature_names[:10])

if __name__ == "__main__":
    main()