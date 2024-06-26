import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """Load Dataset from a JSON file"""
    return pd.read_json(filepath, lines=True)

def explore_data(df):

    """Dataset Description and Overview"""
    print("First few rows of the dataset:\n---------------------------------")
    print(df.head())

    print("\nDataset information:\n---------------------------------")
    print(df.info())

    print("\nSummary statistics:\n---------------------------------")
    print(df.describe())

    print("\nColumns in the dataset:\n---------------------------------")
    print(df.columns.tolist())

    # Sample a few reviews
    print("\nSample reviews:\n---------------------------------")
    print(df['reviewText'].head())
    print("\n---------------------------------")

    # Visualize the distribution of ratings
    plt.figure(figsize=(10,6))
    sns.countplot(x='overall', data=df)
    plt.title('Distribution of Ratings')
    plt.savefig('../notebooks/rating_distribution.png')
    plt.close()

    # Basic text statistics
    df['review_length'] = df['reviewText'].str.len()
    print("\nReview length statistics:")
    print(df['review_length'].describe())

def main():
    print("Loading data...")
    df = load_data("../data/Office_Products_5.json")
    print("Data Loaded.")
    explore_data(df)

if __name__ == "__main__":
    main()