# ============================================================
# ZOMATO RESTAURANT CLUSTERING & SENTIMENT ANALYSIS
# MAIN CONNECTED PIPELINE
# ============================================================

from data_loader import load_data
from data_cleaning import inspect_data, clean_data
from eda import run_eda
from hypothesis_testing import run_hypothesis_tests
from feature_engineering import run_feature_engineering
from clustering_pipeline import run_clustering
from nlp_preprocessing import run_nlp_preprocessing
from sentiment_model import run_sentiment_model

from model_artifacts import (
    save_model_artifacts,
    load_model_artifacts,
    test_loaded_model
)

from mlflow_tracking import (
    setup_mlflow,
    log_sentiment_model
)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():

    print("\n")
    print("=" * 70)
    print("ZOMATO RESTAURANT CLUSTERING & SENTIMENT ANALYSIS")
    print("CONNECTED MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # ========================================================
    # MLFLOW SETUP
    # ========================================================

    setup_mlflow()

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("STEP 1 - LOAD DATA")
    print("=" * 70)

    restaurant_df, review_df = load_data()

    # ========================================================
    # 2. DATA UNDERSTANDING
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("STEP 2 - DATA UNDERSTANDING")
    print("=" * 70)

    inspect_data(
        restaurant_df,
        review_df
    )

    # ========================================================
    # 3. DATA CLEANING
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("STEP 3 - DATA CLEANING")
    print("=" * 70)

    restaurant, review = clean_data(
        restaurant_df,
        review_df
    )

    # ========================================================
    # 4. EDA
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("STEP 4 - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    restaurant, review = run_eda(
        restaurant,
        review
    )

    # ========================================================
    # 5. HYPOTHESIS TESTING
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("STEP 5 - HYPOTHESIS TESTING")
    print("=" * 70)

    hypothesis_results = run_hypothesis_tests(
        restaurant,
        review
    )

    # ========================================================
    # 6. FEATURE ENGINEERING
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("STEP 6 - FEATURE ENGINEERING")
    print("=" * 70)

    feature_results = run_feature_engineering(
        restaurant,
        review
    )

    # ========================================================
    # 7. CLUSTERING
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("STEP 7 - CLUSTERING")
    print("=" * 70)

    clustering_results = run_clustering(
        feature_results,
        review
    )

    # ========================================================
    # 8. NLP PREPROCESSING
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("STEP 8 - NLP PREPROCESSING")
    print("=" * 70)

    nlp_results = run_nlp_preprocessing(
        review
    )

    # ========================================================
    # 9. SENTIMENT MODELING
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("STEP 9 - SENTIMENT MODELING")
    print("=" * 70)

    sentiment_results = run_sentiment_model(
        nlp_results
    )

    # ========================================================
    # 10. MLFLOW TRACKING
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("STEP 10 - MLFLOW TRACKING")
    print("=" * 70)

    mlflow_results = log_sentiment_model(
        sentiment_results
    )

    # ========================================================
    # 11. SAVE MODEL
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("STEP 11 - SAVE MODEL ARTIFACTS")
    print("=" * 70)

    model_path = save_model_artifacts(
        best_model=sentiment_results["best_model"],
        tfidf=sentiment_results["tfidf"]
    )

    # ========================================================
    # 12. LOAD MODEL
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("STEP 12 - LOAD SAVED MODEL")
    print("=" * 70)

    loaded_model, loaded_tfidf = load_model_artifacts(
        model_path
    )

    # ========================================================
    # 13. SANITY CHECK
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("STEP 13 - MODEL SANITY CHECK")
    print("=" * 70)

    test_loaded_model(
        loaded_model,
        loaded_tfidf
    )

    # ========================================================
    # PIPELINE COMPLETE
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "restaurant": restaurant,
        "review": review,
        "hypothesis_results": hypothesis_results,
        "feature_results": feature_results,
        "clustering_results": clustering_results,
        "nlp_results": nlp_results,
        "sentiment_results": sentiment_results,
        "mlflow_results": mlflow_results,
        "model_path": model_path
    }


# ============================================================
# RUN PIPELINE
# ============================================================

if __name__ == "__main__":

    results = run_pipeline()