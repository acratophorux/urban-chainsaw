import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
import joblib
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from wordcloud import WordCloud

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    
def load_tfidf_data():
    """Load TF-IDF data from .npz file"""
    tfidf_matrix = load_npz(os.path.join(PROJECT_DIR, "data", "tfidf", "tfidf_matrix.npz"))
    vectorizer = joblib.load(os.path.join(PROJECT_DIR, "data", "tfidf", 'tfidf_vectorizer.joblib'))
    feature_names = np.load(os.path.join(PROJECT_DIR, "data", "tfidf", 'feature_names.npy'), allow_pickle=True)
    
    return tfidf_matrix, vectorizer, feature_names

def analyze_feature_frequency(tfidf_matrix, feature_names):
    feature_freq = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
    feature_freq = pd.DataFrame({'term':feature_names, 'frequency':feature_freq})
    feature_freq = feature_freq.sort_values('frequency', ascending=False)

    return feature_freq

def plot_top_features(feature_freq, n=20):
    plt.figure(figsize=(12, 6))
    plt.bar(feature_freq['term'][:n], feature_freq['frequency'][:n])
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {n} Most Frequent Terms')
    plt.xlabel('Terms')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, 'data/figures/top_features.png'))
    plt.close()

def plot_tfidf_distribution(tfidf_matrix):
    tfidf_scores = tfidf_matrix.data
    plt.figure(figsize=(10,6))
    plt.hist(tfidf_scores, bins=50, edgecolor='black')
    plt.title('Distribution of TF-IDF Scores')
    plt.xlabel('TF-IDF Score')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(PROJECT_DIR, 'data/figures/tfidf_distribution.png'))
    plt.close()

def perform_topic_modeling(tfidf_matrix, feature_names, n_topics=5, n_top_words=10):
    print("Starting topic modeling...")
    
    # Initialize LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics, 
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Fit the model
    lda.fit(tfidf_matrix)
    
    def get_top_words(topic):
        return [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    
    # Use parallel processing to get top words for each topic
    top_words_list = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(get_top_words)(topic) for topic in lda.components_
    )
    
    # Print results
    for topic_idx, top_words in enumerate(top_words_list):
        print(f"Topic {topic_idx}: {', '.join(top_words)}")
    
    print("Topic modeling completed.")

    return lda


def analyze_sentiment_of_top_terms(df, feature_freq, n=20):
    top_terms = feature_freq['term'][:n].tolist()
    sentiments = []

    for term in top_terms:
        reviews_with_term = df[df['cleaned_review'].str.contains(term, na=False)]
        sentiment = reviews_with_term['overall'].mean()
        sentiments.append(sentiment)
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_terms, sentiments)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Average Rating for Reviews Containing Top {n} Terms')
    plt.xlabel('Terms')
    plt.ylabel('Average Rating')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, 'data/figures/top_terms_sentiment.png'))
    plt.close()

def analyze_term_frequency_by_rating(df, vectorizer):
    ratings = df['overall'].unique()
    for rating in ratings:
        reviews = df[df['overall'] == rating]['cleaned_review']
        tf = vectorizer.transform(reviews)
        tf_sum = tf.sum(axis=0).A1
        top_words = vectorizer.get_feature_names_out()[tf_sum.argsort()[-10:][::-1]]
        print(f"Top 10 words for rating {rating}: {', '.join(top_words)}")
def perform_ngram_analysis(df, n=2):
    ngram_vectorizer = CountVectorizer(ngram_range=(n, n))
    ngrams = ngram_vectorizer.fit_transform(df['cleaned_review'])
    ngram_freq = np.asarray(ngrams.sum(axis=0)).ravel()
    ngram_freq = pd.DataFrame({'ngram':ngram_vectorizer.get_feature_names_out(), 'frequency':ngram_freq})
    print(f"Top 20 {n}-grams")
    print(ngram_freq.head(20))

def main():
    tfidf_matrix, vectorizer, feature_names = load_tfidf_data()

    data_path = os.path.join(PROJECT_DIR, "data", "processed", "processed_luxury_beauty_5.csv")
    df = pd.read_csv(data_path, low_memory=False)
    df['cleaned_review'] = df['cleaned_review'].fillna('')

    feature_freq = analyze_feature_frequency(tfidf_matrix, feature_names)
    plot_top_features(feature_freq)
    plot_tfidf_distribution(tfidf_matrix)
    print(feature_freq.head(20))

    print("\nTopic Modeling Results:")
    lda = perform_topic_modeling(tfidf_matrix, feature_names)

    analyze_sentiment_of_top_terms(df, feature_freq)

    print("\nTerm Frequency by Rating:")
    analyze_term_frequency_by_rating(df, vectorizer)

    perform_ngram_analysis(df)

if __name__ == "__main__":
    main()