# Comment Category Prediction

An NLP-based multi-class classification project for predicting the category
assigned to user-generated comments using text, engagement signals, and
additional metadata.

## Project Links

- **GitHub:** [Source code and project files](https://github.com/Sanandha1234/Comment-category-prediction)
- **Streamlit Demo:** [Try the deployed app](https://cdax5drf3njr8ag9ye4raq.streamlit.app/)
- **Kaggle Notebook:** [View notebook](https://www.kaggle.com/code/psanandha/24f2008134-notebook-t12026) *(private; access may be restricted)*

## Overview

This project was developed as part of the **Comment Category Prediction
Challenge** on Kaggle.

The objective is to predict the final category assigned to a comment using
a combination of:

- Comment text
- Engagement features
- Emoticon indicators
- Topic-related indicators
- Temporal features
- Internal system features

The project combines exploratory data analysis, feature engineering, NLP
techniques, and machine learning models.

## Dataset

The competition dataset contains:

- Training data
- Test data
- Sample submission file

The main features include:

| Feature | Description |
|---|---|
| `comment` | Raw comment text |
| `created_date` | Date and time of the comment |
| `post_id` | Discussion/post identifier |
| `upvote` | Number of positive reactions |
| `downvote` | Number of negative reactions |
| `emoticon_1` | Emoticon indicator |
| `emoticon_2` | Emoticon indicator |
| `emoticon_3` | Emoticon indicator |
| `if_1` | Internal system feature |
| `if_2` | Internal system feature |
| `race` | Topic indicator |
| `religion` | Topic indicator |
| `gender` | Topic indicator |
| `disability` | Topic indicator |
| `label` | Target variable |

## Methodology

### 1. Exploratory Data Analysis

The dataset was analyzed to understand:

- Dataset dimensions
- Data types
- Missing values
- Target distribution
- Numerical features
- Comment characteristics
- Outliers

### 2. Data Cleaning

Missing values were handled and the data was prepared for machine learning.

### 3. Feature Engineering

Additional features were created from the comments and engagement signals,
including:

- Comment length
- Word count
- Exclamation count
- Question count
- Capitalization ratio
- Total emoticons
- Vote difference
- Total votes

### 4. Text Feature Engineering

TF-IDF was used to transform the comment text into numerical features.

Both word-level and character-level TF-IDF were explored to capture
different textual patterns.

### 5. Machine Learning Models

Three models were evaluated:

- LinearSVC
- Logistic Regression
- LightGBM

Hyperparameter tuning was also performed.

## Model Performance

| Model | Accuracy | F1 Score |
|---|---:|---:|
| LinearSVC | 78.79% | 78.31% |
| Logistic Regression | 82.55% | 78.56% |
| **LightGBM** | **84.67%** | **82.21%** |

### Best Model

**LightGBM**

- Validation Accuracy: **84.67%**
- Validation F1 Score: **82.21%**

LightGBM achieved the best validation performance among the evaluated
models.

## Key Findings

- TF-IDF provided an effective representation of comment text.
- Character-level features helped capture additional textual patterns.
- Feature engineering provided useful information beyond raw text.
- Logistic Regression provided a strong baseline.
- LightGBM achieved the best overall validation performance.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- SciPy
- Matplotlib
- TF-IDF
- Natural Language Processing
- Machine Learning

## Project Structure

```text
comment-category-prediction/
│
├── README.md
├── comment-category-prediction.ipynb
├── requirements.txt
├── results/
│   ├── model_comparison.png
│   └── confusion_matrix.png
└── .gitignore
