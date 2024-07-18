# NLP Project: Sentiment Analysis and Topic Classification for Amazon Office Product Reviews

## Project Overview

This project aims to perform sentiment analysis and topic classification on Amazon Luxury Beauty Products' reviews using natural language processing (NLP) techniques.

## Dataset Overview

We are using the Amazon Luxury Beauty Products 5-core dataset, which includes reviews for luxury beauty products sold on Amazon.com.

Key characteristics of the dataset:

- Total number of reviews: 34278
- Features include: overall rating, verified purchase status, review text, summary, and timestamps. Of these, only rating, review text, and summary features are used in this project.

Dataset statistics:

- Ratings distribution:
  - Mean rating: 4.28616 (out of 5)
- Review length:
  - Mean: 488 characters
  - Min: 0 character
  - Max: 16,970 characters

Here is an overview of dataset:

```
Dataset shape: (34278, 6)

Column names: ['overall', 'reviewText', 'summary', 'cleaned_review', 'cleaned_summary', 'review_length']

Sample preprocessed reviews:
0    handcream beauti fragranc doesnt stay protect ...
1    wonder hand lotion serious dri skin stay long ...
2    best hand cream around silki thick soak way le...
3                                                thank
4    great hand lotion soak right leav skin super s...
Name: cleaned_review, dtype: object

Distribution of ratings:
overall
1     1095
2     1496
3     3884
4     7833
5    19970
Name: count, dtype: int64

Review length statistics:
count    34278.000000
mean       488.958516
std        605.822360
min          0.000000
25%        116.000000
50%        321.000000
75%        672.000000
max      16970.000000
Name: review_length, dtype: float64
```

## Progress Journal

### Wed Jun 26

- Set up project structure and environment
- Chose the Office Products dataset from Amazon reviews
- Implemented initial data exploration script
- Began preprocessing steps including text cleaning, tokenization, and feature engineering
- Added dataset overview to project documentation

### Fri Jun 28

- Completed data preprocessing, including improved regex for text cleaning
- Implemented TF-IDF feature extraction
- Saved TF-IDF matrix, vectorizer, and feature names for future use

### Tue Jul 16

- Switched dataset from Office Products to Luxury Beauty Products due to the large size of Office Products dataset and limited compute resources
- Caught upto dataset loading, preprocessing, and eda

### Thu Jul 18

- Performed initial Exploratory Data Analysis (EDA) on the preprocessed Luxury Beauty dataset
- Analyzed most frequent terms in the reviews
- Conducted topic modeling using Latent Dirichlet Allocation (LDA), identifying 5 distinct topics
- Examined term frequency distribution across different rating levels
- Attempted initial n-gram analysis

Key findings:

1. Most frequent terms include "color", "product", "use", "love", and "skin", aligning well with beauty product reviews
2. LDA revealed topics related to skincare, fragrances, nail care, and general hair/skin products
3. Observed clear differences in term usage across different ratings
4. Identified potential areas for improvement in n-gram analysis

### Next Steps

- Modeling
- Evaluation and error analysis

## Project Structure

```
root/
│
├── data/               # For storing datasets
│   ├── Luxury_Beauty_5.json
|   ├── figures/
|   └── processed/
│       └── processed_luxury_beauty_5.csv
├── src/                # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py
│   └── feature_extraction.py
├── notebooks/          # For Jupyter notebooks (not used)
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Setup and Installation

[Include instructions on how to set up the project environment]

## Usage

[Include instructions on how to run your scripts]

## Results and Findings

[To be updated as the project progresses]

## Future Improvements

[To be updated as the project progresses]

## References

- [List any references or resources used]

```

```
