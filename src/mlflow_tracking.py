# ============================================================
# ZOMATO RESTAURANT CLUSTERING & SENTIMENT ANALYSIS
# MLFLOW TRACKING
# ============================================================

import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# MLFLOW SETTINGS
# ============================================================

EXPERIMENT_NAME = "Zomato Sentiment Analysis"

TRACKING_URI = "sqlite:///mlflow.db"


# ============================================================
# SETUP MLFLOW
# ============================================================

def setup_mlflow():

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print("\nMLflow Tracking URI:")
    print(
        mlflow.get_tracking_uri()
    )

    print("\nMLflow Experiment:")
    print(
        EXPERIMENT_NAME
    )


# ============================================================
# LOG SENTIMENT MODEL
# ============================================================

def log_sentiment_model(
    sentiment_results
):

    model = sentiment_results["best_model"]

    tfidf = sentiment_results["tfidf"]

    X_test = sentiment_results["X_test"]

    y_test = sentiment_results["y_test"]

    best_model_name = sentiment_results[
        "best_model_name"
    ]

    # ========================================================
    # TRANSFORM TEST DATA
    # ========================================================

    X_test_tfidf = tfidf.transform(
        X_test
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    predictions = model.predict(
        X_test_tfidf
    )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label="Positive"
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="Positive"
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="Positive"
    )

    # ========================================================
    # START MLFLOW RUN
    # ========================================================

    with mlflow.start_run(
        run_name=best_model_name
    ):

        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        mlflow.log_param(
            "model_name",
            best_model_name
        )

        mlflow.log_param(
            "model_type",
            type(model).__name__
        )

        mlflow.log_param(
            "tfidf_max_features",
            tfidf.max_features
        )

        mlflow.log_param(
            "test_samples",
            len(X_test)
        )

        # ----------------------------------------------------
        # MODEL PARAMETERS
        # ----------------------------------------------------

        model_params = model.get_params()

        for parameter, value in model_params.items():

            try:

                mlflow.log_param(
                    parameter,
                    value
                )

            except Exception:

                pass

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # ----------------------------------------------------
        # LOG SENTIMENT MODEL
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="sentiment_model"
        )

        # ----------------------------------------------------
        # SAVE TF-IDF TEMPORARILY
        # ----------------------------------------------------

        tfidf_path = "tfidf_vectorizer.pkl"

        joblib.dump(
            tfidf,
            tfidf_path
        )

        # ----------------------------------------------------
        # LOG TF-IDF
        # ----------------------------------------------------

        mlflow.log_artifact(
            tfidf_path
        )

        # ----------------------------------------------------
        # REMOVE TEMPORARY FILE
        # ----------------------------------------------------

        if os.path.exists(
            tfidf_path
        ):

            os.remove(
                tfidf_path
            )

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("MLFLOW RUN COMPLETED")
        print("=" * 70)

        print(
            "\nModel:",
            best_model_name
        )

        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Precision: {precision:.4f}"
        )

        print(
            f"Recall   : {recall:.4f}"
        )

        print(
            f"F1 Score : {f1:.4f}"
        )

    # ========================================================
    # RETURN MLFLOW METRICS
    # ========================================================

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }