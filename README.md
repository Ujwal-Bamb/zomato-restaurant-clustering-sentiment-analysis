# Zomato Restaurant Clustering & Sentiment Analysis

## Project Overview

This project analyzes Zomato restaurant and customer review data using
Exploratory Data Analysis, statistical hypothesis testing, unsupervised
learning, and NLP-based sentiment classification.

## Objectives

- Analyze restaurant pricing and cuisine patterns
- Identify restaurant segments using clustering
- Analyze customer ratings and reviews
- Perform sentiment classification on customer reviews
- Identify important words influencing sentiment predictions

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- NLTK
- SHAP

## Machine Learning Techniques

### Unsupervised Learning

- K-Means Clustering
- Agglomerative Hierarchical Clustering
- PCA
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

### NLP & Supervised Learning

- Text preprocessing
- Tokenization
- Lemmatization
- TF-IDF
- Logistic Regression
- Naive Bayes
- Linear SVM
- Hyperparameter Tuning
- SHAP Explainability

## Key Findings

- Restaurant cost and customer rating showed a significant positive association.
- Cuisine variety showed a positive association with restaurant cost.
- Approximately 63% of rated reviews were classified as Positive based on
  the project's rating-derived sentiment definition.
- Restaurant clustering identified groups based on pricing and cuisine
  characteristics.
- Tuned Logistic Regression achieved approximately 89.86% F1 Score on
  the test dataset.
- SHAP analysis identified influential words used by the sentiment model.

## Conclusion

This project analyzed Zomato restaurant and customer review data to
understand restaurant characteristics, identify restaurant segments,
and analyze customer sentiment.

The analysis found significant positive associations between restaurant
cost and customer ratings, as well as between cuisine variety and cost.
Clustering was used to group restaurants based on their pricing and
cuisine characteristics, although the relatively low cluster separation
suggests that the segments should be interpreted carefully.

For customer review analysis, NLP techniques and TF-IDF were used to
classify reviews into Positive and Negative sentiment based on
rating-derived labels. Among the evaluated models, Tuned Logistic
Regression achieved the highest F1 Score of approximately 89.86%, with
90.33% Precision and 89.39% Recall on the test set.

SHAP analysis provided additional interpretability by identifying
important text features influencing the model's predictions.

Overall, the project demonstrates how EDA, statistical analysis,
unsupervised learning, and NLP-based supervised learning can be combined
to derive useful insights from restaurant and customer review data.

## Limitations

- Small number of restaurants available for clustering
- Sentiment labels are derived from ratings rather than independent
  human annotations
- Limited clustering features
- Relatively weak cluster separation
- Results are specific to the available dataset

## Future Scope

- Use a larger and more recent dataset
- Add more restaurant-level features
- Perform aspect-based sentiment analysis
- Build a restaurant recommendation system
- Develop an interactive Streamlit application
- Deploy the model using FastAPI or cloud infrastructure