import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    return pd.read_csv(filepath)

def explore_data(df):

    """Dataset Description and Overview"""
    print("First few rows of the dataset:\n---------------------------------")
    print(df.head())

    print("\nDataset information:\n---------------------------------")
    print(df.info())

    print("\nSummary statistics:\n---------------------------------")
    print(df.describe())

    print("Columns in the dataset:\n---------------------------------")
    print(df.columns)
    print("\n---------------------------------")
    # Visualize the distribution of ratings
    plt.figure(figsize=(10,6))
    sns.countplot(x='rating', data=df)
    plt.title('Distribution of Ratings')
    plt.savefig('../notebooks/rating_distribution.png')
    plt.close()

def main():
    df = load_data("../data/amazon_reviews.csv")

    explore_data(df)

if __name__ == "__main__":
    main()