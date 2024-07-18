import os
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    
def load_processed_data():
    """Load processed data from csv file"""

    df = pd.read_csv(os.path.join(PROJECT_DIR, "data/processed/processed_luxury_beauty_5.csv"),
                     low_memory=False)
    df['cleaned_review'] = df['cleaned_review'].fillna('')
    return df


def load_tfidf_data():
    """Load TF-IDF data from .npz file"""
    tfidf_matrix = load_npz(os.path.join(PROJECT_DIR, "data", "tfidf", "tfidf_matrix.npz"))
    vectorizer = joblib.load(os.path.join(PROJECT_DIR, "data", "tfidf", 'tfidf_vectorizer.joblib'))
    feature_names = np.load(os.path.join(PROJECT_DIR, "data", "tfidf", 'feature_names.npy'), allow_pickle=True)
    
    return tfidf_matrix, vectorizer, feature_names

def create_sentiment_labels(df):
    """Create sentiment labels based on overall rating
            4 or 5: positive
            3: neutral
            1 or 2: negative
    """
    def to_sentiment(rating):
        if rating <= 2:
            return 'Negative'
        elif rating == 3:
            return 'Neutral'
        else:
            return 'Positive'

    df['sentiment'] = df['overall'].apply(to_sentiment)
    return df

def model(X_train, y_train, X_test, y_test, le):

    # Initialize and train the model
    model = LogisticRegression(random_state=42, max_iter=2000)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Convert numeric predictions back to string labels
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)

    # Evaluate the model
    print(classification_report(y_test_labels, y_pred_labels))

def main():

    df = load_processed_data()
    tfidf_matrix, vectorizer, feature_names = load_tfidf_data()
    
    # features from the tfidf-matrix
    X = tfidf_matrix

    # labels
    df = create_sentiment_labels(df)
    y = df['sentiment']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    print("Training set shape:", X_train.shape)
    print("Testing set shape:", X_test.shape)
    print("Number of classes:", len(np.unique(y_encoded)))

    # train a simple logistic regression model
    print("Training logistic regression model...")
    model(X_train, y_train, X_test, y_test, le)
    print("Done.")

if __name__ == "__main__":
    main()