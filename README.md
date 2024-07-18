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

# Logistic Regression Model Performance Analysis

## Model Details

- Algorithm: Logistic Regression
- Training set shape: (27422, 5000)
- Testing set shape: (6856, 5000)
- Number of classes: 3

## Performance Metrics

|              | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Negative     | 0.77      | 0.38   | 0.51     | 513     |
| Neutral      | 0.59      | 0.27   | 0.38     | 745     |
| Positive     | 0.88      | 0.98   | 0.93     | 5598    |
| Accuracy     |           |        | 0.86     | 6856    |
| Macro Avg    | 0.75      | 0.54   | 0.60     | 6856    |
| Weighted Avg | 0.84      | 0.86   | 0.84     | 6856    |

## Analysis

1. **Overall Performance**: The model achieves an accuracy of 86%, which is good. However, there's room for improvement, especially in handling negative and neutral sentiments.

2. **Class Imbalance**: There's a significant class imbalance in the dataset. Positive reviews (5598) far outnumber negative (513) and neutral (745) reviews.

3. **Positive Sentiment**: The model performs exceptionally well on positive reviews (F1-score: 0.93, Recall: 0.98). This is likely due to the abundance of positive samples in the dataset.

4. **Negative and Neutral Sentiments**: The model struggles with negative and neutral reviews, particularly in terms of recall (0.38 and 0.27 respectively). This means it's missing a lot of negative and neutral reviews.

5. **Precision vs Recall**: For negative and neutral classes, precision is significantly higher than recall. This suggests the model is cautious about labeling reviews as negative or neutral, but when it does, it's often correct.

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

## Project Structure

```
root/
│
├── data/
│   ├── Luxury_Beauty_5.json
|   ├── figures/
|   ├── processed/
│   │    └── processed_luxury_beauty_5.csv
|   └── tfidf/
│       ├── feature_names.npy
│       ├── tfidf_matrix.npz
│       └── tfidf_vectorizer.joblib
├── notebooks/          # For Jupyter notebooks (not used)
├── src/                # Source code
│   ├── __init__.py
│   ├── app.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   └── feature_extraction.py
│   └── model.py
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
