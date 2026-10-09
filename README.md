## Project Overview

This project analyzes Zomato restaurant data and customer reviews using
machine learning, clustering, natural language processing, and sentiment
classification.

The project has two major components:

1. Restaurant clustering
2. Customer review sentiment analysis

Restaurant metadata was analyzed to identify groups of restaurants based
on characteristics such as cost, cuisine count, ratings, and other
restaurant features.

Customer reviews were processed using NLP techniques and TF-IDF features,
followed by supervised sentiment classification.

The final sentiment model is a tuned Logistic Regression model, which was
tracked using MLflow and deployed through a Streamlit application.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- TF-IDF
- K-Means Clustering
- Agglomerative Clustering
- Logistic Regression
- Multinomial Naive Bayes
- Linear SVC
- SHAP
- MLflow
- Joblib
- Streamlit
- Git/GitHub

## Machine Learning Workflow

1. Data Cleaning
2. Exploratory Data Analysis
3. Hypothesis Testing
4. Feature Engineering
5. Restaurant Clustering
6. NLP Preprocessing
7. TF-IDF Feature Extraction
8. Sentiment Classification
9. Hyperparameter Tuning
10. SHAP Explainability
11. Model Saving
12. MLflow Experiment Tracking
13. Streamlit Deployment

## Final Sentiment Model

The final model selected for deployment is:

**Tuned Logistic Regression**

Performance:

- Accuracy: approximately 87%
- Precision: 90.33%
- Recall: 89.39%
- F1 Score: approximately 89.86%

## Deployment

The trained model and TF-IDF vectorizer are saved using Joblib.

A Streamlit application allows users to enter a restaurant review and
receive a Positive or Negative sentiment prediction.

## Experiment Tracking

MLflow was used to track:

- Model parameters
- Accuracy
- Precision
- Recall
- F1 Score
- Model artifacts
- TF-IDF vectorizer

## Project Structure

```text
data/
models/
src/
app.py
module_6_ML_Submission_project.ipynb
requirements.txt
README.md
