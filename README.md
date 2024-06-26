# NLP Project: Sentiment Analysis and Topic Classification for Amazon Office Product Reviews

## Project Overview

This project aims to perform sentiment analysis and topic classification on Amazon Office Product reviews using natural language processing (NLP) techniques.

## Dataset Overview

We are using the Amazon Office Products 5-core dataset, which includes reviews for office products sold on Amazon.com.

Key characteristics of the dataset:

- Total number of reviews: 800,357
- Time span: [TODO: add this based on the min and max of unixReviewTime]
- Features include: overall rating, verified purchase status, review text, summary, and timestamps

Dataset statistics:

- Ratings distribution:
  - Mean rating: 4.47 (out of 5)
  - Median rating: 5.0
- Review length:
  - Mean: 241 characters
  - Median: 108 characters
  - Min: 1 character
  - Max: 32,602 characters

Notable columns:

- 'overall': The product's rating (1 to 5)
- 'verified': Boolean indicating if the review is from a verified purchase
- 'reviewText': The text of the review
- 'summary': A summary of the review
- 'unixReviewTime': The time of the review in Unix time

## Progress Journal

### Wed Jun 26

- Set up project structure and environment
- Chose the Office Products dataset from Amazon reviews
- Implemented initial data exploration script
- Began preprocessing steps including text cleaning, tokenization, and feature engineering

### Next steps

- Complete data preprocessing
- Implement feature extraction (e.g., TF-IDF)
- Split data into training and testing sets
- Begin model selection and training

## Project Structure

```
project_folder/
│
├── data/               # For storing datasets
├── src/                # Source code
│   ├── __init__.py
│   ├── data_preprocessing.py
│   └── [future scripts]
├── notebooks/          # For Jupyter notebooks (if used)
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
