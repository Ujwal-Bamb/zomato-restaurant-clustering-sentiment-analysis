# ============================================================
# ZOMATO RESTAURANT CLUSTERING & SENTIMENT ANALYSIS
# MODEL ARTIFACTS
# ============================================================

import os
import joblib


# ============================================================
# FILE PATHS
# ============================================================

SENTIMENT_MODEL_PATH = "models/zomato_sentiment_model.joblib"

CLUSTERING_ARTIFACTS_PATH = (
    "models/zomato_clustering_artifacts.joblib"
)


# ============================================================
# SENTIMENT MODEL
# ============================================================

def save_model_artifacts(
    best_model,
    tfidf,
    model_path=SENTIMENT_MODEL_PATH
):
    """
    Save the trained sentiment model and TF-IDF vectorizer.
    """

    os.makedirs(
        os.path.dirname(model_path),
        exist_ok=True
    )

    model_package = {
        "model": best_model,
        "tfidf": tfidf
    }

    joblib.dump(
        model_package,
        model_path
    )

    print("\n" + "=" * 70)
    print("SENTIMENT MODEL SAVED")
    print("=" * 70)

    print(
        f"Model saved successfully to: {model_path}"
    )

    return model_path


def load_model_artifacts(
    model_path=SENTIMENT_MODEL_PATH
):
    """
    Load the saved sentiment model and TF-IDF vectorizer.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Sentiment model not found at: {model_path}"
        )

    loaded_package = joblib.load(
        model_path
    )

    model = loaded_package["model"]
    tfidf = loaded_package["tfidf"]

    return model, tfidf


# ============================================================
# TEST SAVED SENTIMENT MODEL
# ============================================================

def test_loaded_model(
    model_path=SENTIMENT_MODEL_PATH
):
    """
    Test the saved sentiment model
    using sample reviews.
    """

    model, tfidf = load_model_artifacts(
        model_path
    )

    sample_reviews = [
        "The food was absolutely delicious and the service was excellent.",
        "Very bad experience. The food was cold and the service was terrible."
    ]

    sample_vectors = tfidf.transform(
        sample_reviews
    )

    predictions = model.predict(
        sample_vectors
    )

    print("\n" + "=" * 70)
    print("LOADED MODEL TEST")
    print("=" * 70)

    for review, prediction in zip(
        sample_reviews,
        predictions
    ):
        print("\nReview:")
        print(review)

        print("Prediction:")
        print(prediction)

    print(
        "\nNumber of TF-IDF features:",
        len(tfidf.get_feature_names_out())
    )

    print(
        "Model type:",
        type(model).__name__
    )

    return predictions


# ============================================================
# CLUSTERING ARTIFACTS
# ============================================================

def save_clustering_artifacts(
    clustering_results,
    artifact_path=CLUSTERING_ARTIFACTS_PATH
):
    """
    Save clustering outputs required by Streamlit.

    The clustering pipeline already returns:
        - restaurant
        - kmeans_model
        - hierarchical_model
        - kmeans_labels
        - hierarchical_labels
        - kmeans_profile
        - hierarchical_profile
        - comparison

    We save the complete clustering result so that
    Streamlit can load the results without retraining.
    """

    os.makedirs(
        os.path.dirname(artifact_path),
        exist_ok=True
    )

    clustering_package = {
        "restaurant": clustering_results["restaurant"],
        "kmeans_model": clustering_results["kmeans_model"],
        "hierarchical_model": clustering_results["hierarchical_model"],
        "kmeans_labels": clustering_results["kmeans_labels"],
        "hierarchical_labels": clustering_results["hierarchical_labels"],
        "kmeans_profile": clustering_results["kmeans_profile"],
        "hierarchical_profile": clustering_results["hierarchical_profile"],
        "comparison": clustering_results["comparison"]
    }

    joblib.dump(
        clustering_package,
        artifact_path
    )

    print("\n" + "=" * 70)
    print("CLUSTERING ARTIFACTS SAVED")
    print("=" * 70)

    print(
        f"Clustering artifacts saved successfully to: "
        f"{artifact_path}"
    )

    return artifact_path


# ============================================================
# LOAD CLUSTERING ARTIFACTS
# ============================================================

def load_clustering_artifacts(
    artifact_path=CLUSTERING_ARTIFACTS_PATH
):
    """
    Load saved clustering artifacts.
    """

    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Clustering artifacts not found at: "
            f"{artifact_path}"
        )

    clustering_package = joblib.load(
        artifact_path
    )

    return clustering_package


# ============================================================
# GET RESTAURANT DATA FROM CLUSTERING ARTIFACTS
# ============================================================

def load_restaurant_data(
    artifact_path=CLUSTERING_ARTIFACTS_PATH
):
    """
    Load only the restaurant dataframe.
    """

    clustering_package = load_clustering_artifacts(
        artifact_path
    )

    return clustering_package["restaurant"]


# ============================================================
# GET K-MEANS PROFILE
# ============================================================

def load_kmeans_profile(
    artifact_path=CLUSTERING_ARTIFACTS_PATH
):
    """
    Load K-Means cluster profiles.
    """

    clustering_package = load_clustering_artifacts(
        artifact_path
    )

    return clustering_package["kmeans_profile"]


# ============================================================
# GET CLUSTERING COMPARISON
# ============================================================

def load_clustering_comparison(
    artifact_path=CLUSTERING_ARTIFACTS_PATH
):
    """
    Load K-Means vs Agglomerative comparison.
    """

    clustering_package = load_clustering_artifacts(
        artifact_path
    )

    return clustering_package["comparison"]