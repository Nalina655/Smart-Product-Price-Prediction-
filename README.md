# Smart-Product-Price-Prediction-
Hackathon project
💰 Smart Product Price Prediction using LightGBM
🧠 ML Challenge 2025 — Smart Product Pricing Challenge
📖 Overview

In modern e-commerce, determining the optimal price point for products is crucial for competitiveness and customer satisfaction. This project, developed as part of the ML Challenge 2025, focuses on predicting product prices using machine learning by analyzing product descriptions, specifications, and structured metadata.

The model leverages text-based catalog data (titles, descriptions, item pack quantities) and applies advanced NLP feature engineering techniques to capture semantic and structural information. A LightGBM regression model, optimized with cross-validation, was trained to predict product prices with high accuracy and generalization capability.

🧠 Problem Statement

Develop an ML model that predicts the price of a product based on its catalog content (text) and other details such as brand and quantity.
The goal is to minimize SMAPE (Symmetric Mean Absolute Percentage Error) between predicted and actual prices.

⚙️ Key Features

🏷️ Predicts optimal product prices from catalog text data.

🔍 Extracts structured information (brand, quantity, volume) using NLP and regex-based feature engineering.

💡 Applies TF-IDF + SVD for text dimensionality reduction.

⚡ Trains a LightGBM Regressor with tuned hyperparameters and 7-fold cross-validation.

📊 Achieved approximately 50% SMAPE on validation data.

🧰 Tools & Technologies

Programming Language: Python

Libraries/Frameworks:

Pandas, NumPy — Data Handling

Scikit-learn — Feature Engineering & Evaluation

LightGBM — Gradient Boosting Model

NLTK / Regex — Text Cleaning & Tokenization

TF-IDF + SVD — NLP Feature Extraction & Dimensionality Reduction

Development Environment: Jupyter Notebook / VS Code

📂 Project Structure
SmartProductPricePrediction/
│
├── dataset/
│   ├── train.csv               # Training data (with prices)
│   ├── test.csv                # Test data (without prices)
│   └── sample_test_out.csv     # Sample output format
│
├── src/
│   ├── preprocess.py           # Text cleaning & feature extraction
│   ├── model_train.py          # Model training & evaluation script
│   ├── predict.py              # Generates final predictions
│   ├── utils.py                # Utility functions (e.g., download images)
│   └── test.ipynb              # Notebook for testing the workflow
│
├── outputs/
│   └── test_out.csv            # Final submission file
│
└── README.md

🧩 Methodology
1. Data Preprocessing

Cleaned catalog text: removed punctuation, stopwords, and lowercased text.

Extracted structured features:

Brand name

Item Pack Quantity (IPQ)

Volume / Weight keywords (ml, g, L, etc.)

Handled outliers and missing values.

2. Feature Engineering

Text Representation:

Applied TF-IDF vectorization on cleaned catalog text.

Reduced dimensionality using Truncated SVD (Latent Semantic Analysis).

Structured Features:

Concatenated TF-IDF components with numeric attributes (quantity, brand frequency, etc.).

3. Modeling

Model used: LightGBM Regressor

7-fold Cross-Validation for robust performance estimation.

Hyperparameter tuning for:

Learning rate, max depth, num_leaves, and feature fraction.

4. Evaluation

Metric: SMAPE (Symmetric Mean Absolute Percentage Error)

𝑆
𝑀
𝐴
𝑃
𝐸
=
1
𝑛
∑
∣
𝑃
𝑟
𝑒
𝑑
𝑖
𝑐
𝑡
𝑒
𝑑
−
𝐴
𝑐
𝑡
𝑢
𝑎
𝑙
∣
(
∣
𝑃
𝑟
𝑒
𝑑
𝑖
𝑐
𝑡
𝑒
𝑑
∣
+
∣
𝐴
𝑐
𝑡
𝑢
𝑎
𝑙
∣
)
/
2
SMAPE=
n
1
	​

∑
(∣Predicted∣+∣Actual∣)/2
∣Predicted−Actual∣
	​


Achieved ~50% SMAPE on validation data.

🧪 Example Code Snippet
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load data
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

# TF-IDF on catalog content
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_tfidf = tfidf.fit_transform(train['catalog_content'])

# Dimensionality reduction
svd = TruncatedSVD(n_components=100, random_state=42)
X_svd = svd.fit_transform(X_tfidf)

# Prepare model input
X = np.hstack([X_svd])
y = train['price']

# Model training
model = LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=64)
kf = KFold(n_splits=7, shuffle=True, random_state=42)

preds = np.zeros(len(test))
for train_idx, val_idx in kf.split(X):
    model.fit(X[train_idx], y.iloc[train_idx])
    preds += model.predict(svd.transform(tfidf.transform(test['catalog_content']))) / kf.n_splits

# Output file
submission = pd.DataFrame({'sample_id': test['sample_id'], 'price': preds})
submission.to_csv('outputs/test_out.csv', index=False)

📈 Results
Metric	Value
SMAPE (Validation)	~50%
Model	LightGBM Regressor
Feature Representation	TF-IDF + SVD
Cross Validation	7-Fold
🚀 Future Enhancements

Incorporate image-based embeddings using CNN or CLIP for multimodal learning.

Experiment with Transformer-based text encoders (BERT, RoBERTa).

Deploy as an interactive Streamlit or Gradio app for real-time product price estimation.

🧑‍💻 Developer

Nalina S D
🎓 Final-year ECE Student, PES University
📊 Project: Smart Product Price Prediction using LightGBM
📅 ML Challenge 2025 Submission
