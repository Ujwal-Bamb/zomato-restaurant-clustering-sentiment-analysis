# ============================================================
# ZOMATO RESTAURANT CLUSTERING & SENTIMENT ANALYSIS
# MODEL ARTIFACTS
# ============================================================

import os
import joblib


MODEL_PATH = "models/zomato_sentiment_model.joblib"


def save_model_artifacts(
    best_model,
    tfidf,
    model_path=MODEL_PATH
):

    print("\n" + "=" * 70)
    print("SAVING MODEL ARTIFACTS")
    print("=" * 70)

    # ========================================================
    # CREATE MODELS DIRECTORY
    # ========================================================

    os.makedirs(
        os.path.dirname(model_path),
        exist_ok=True
    )

    # ========================================================
    # CREATE MODEL PACKAGE
    # ========================================================

    model_package = {
        "model": best_model,
        "tfidf": tfidf
    }

    # ========================================================
    # SAVE MODEL PACKAGE
    # ========================================================

    joblib.dump(
        model_package,
        model_path
    )

    print(
        "Sentiment model and TF-IDF vectorizer "
        "saved successfully."
    )

    print(
        f"Saved to: {model_path}"
    )

    return model_path


def load_model_artifacts(
    model_path=MODEL_PATH
):

    print("\n" + "=" * 70)
    print("LOADING MODEL ARTIFACTS")
    print("=" * 70)

    loaded_package = joblib.load(
        model_path
    )

    loaded_model = loaded_package["model"]

    loaded_tfidf = loaded_package["tfidf"]

    print(
        "Model loaded successfully."
    )

    return loaded_model, loaded_tfidf


def test_loaded_model(
    loaded_model,
    loaded_tfidf
):

    print("\n" + "=" * 70)
    print("MODEL SANITY CHECK")
    print("=" * 70)

    # ========================================================
    # SAMPLE REVIEWS
    # ========================================================

    sample_reviews = [
        "The food was excellent and the service was amazing.",
        "The food was terrible and the service was very slow."
    ]

    # ========================================================
    # TRANSFORM NEW REVIEWS
    # USING THE SAVED TF-IDF VECTORIZER
    # ========================================================

    sample_vectors = loaded_tfidf.transform(
        sample_reviews
    )

    # ========================================================
    # PREDICT SENTIMENT
    # ========================================================

    predictions = loaded_model.predict(
        sample_vectors
    )

    # ========================================================
    # DISPLAY PREDICTIONS
    # ========================================================

    for review_text, prediction in zip(
        sample_reviews,
        predictions
    ):

        print("Review:", review_text)
        print("Prediction:", prediction)
        print("-" * 70)

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    print(
        "Number of TF-IDF features:",
        len(
            loaded_tfidf
            .get_feature_names_out()
        )
    )

    print(
        "Model loaded:",
        type(loaded_model).__name__
    )