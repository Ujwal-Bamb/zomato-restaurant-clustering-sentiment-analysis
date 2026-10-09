# ============================================================
# ZOMATO RESTAURANT CLUSTERING & SENTIMENT ANALYSIS
# SENTIMENT MODEL
# NOTEBOOK-ALIGNED VERSION
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    make_scorer
)


# ============================================================
# MAIN FUNCTION
# ============================================================

def run_sentiment_model(nlp_results):

    print("\n" + "=" * 70)
    print("SENTIMENT MODELING")
    print("=" * 70)

    X = nlp_results["X"]
    y = nlp_results["y"]

    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples :", len(X_test))

    # ========================================================
    # TF-IDF
    # ========================================================

    tfidf = TfidfVectorizer(
        max_features=5000
    )

    X_train_tfidf = tfidf.fit_transform(
        X_train
    )

    X_test_tfidf = tfidf.transform(
        X_test
    )

    print("\nTF-IDF training shape:")
    print(X_train_tfidf.shape)

    print("\nTF-IDF testing shape:")
    print(X_test_tfidf.shape)

    # ========================================================
    # F1 SCORER
    # IMPORTANT:
    # MATCHES ORIGINAL NOTEBOOK
    # ========================================================

    f1_scorer = make_scorer(
        f1_score,
        pos_label="Positive"
    )

    # ========================================================
    # HELPER FUNCTION FOR MODEL EVALUATION
    # ========================================================

    def evaluate_model(
        model,
        X_data,
        y_data,
        model_name
    ):

        predictions = model.predict(
            X_data
        )

        accuracy = accuracy_score(
            y_data,
            predictions
        )

        precision = precision_score(
            y_data,
            predictions,
            pos_label="Positive"
        )

        recall = recall_score(
            y_data,
            predictions,
            pos_label="Positive"
        )

        f1 = f1_score(
            y_data,
            predictions,
            pos_label="Positive"
        )

        print("\n" + "-" * 60)
        print(model_name)
        print("-" * 60)

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

        print("\nClassification Report:")

        print(
            classification_report(
                y_data,
                predictions
            )
        )

        print("\nConfusion Matrix:")

        print(
            confusion_matrix(
                y_data,
                predictions
            )
        )

        return {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }

    # ========================================================
    # 1. LOGISTIC REGRESSION - BASELINE
    # ========================================================

    print("\n" + "=" * 70)
    print("1. LOGISTIC REGRESSION")
    print("=" * 70)

    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    lr_model.fit(
        X_train_tfidf,
        y_train
    )

    lr_baseline = evaluate_model(
        lr_model,
        X_test_tfidf,
        y_test,
        "Logistic Regression - Baseline"
    )

    # ========================================================
    # LOGISTIC REGRESSION - GRID SEARCH
    # ========================================================

    lr_param_grid = {
        "C": [
            0.01,
            0.1,
            1,
            10,
            100
        ],
        "solver": [
            "liblinear",
            "lbfgs"
        ]
    }

    lr_grid = GridSearchCV(
        estimator=LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ),
        param_grid=lr_param_grid,
        scoring=f1_scorer,
        cv=5,
        n_jobs=1
    )

    lr_grid.fit(
        X_train_tfidf,
        y_train
    )

    best_lr = lr_grid.best_estimator_

    print("\nBest Logistic Regression parameters:")
    print(
        lr_grid.best_params_
    )

    print(
        f"Best Logistic Regression CV F1: "
        f"{lr_grid.best_score_:.4f}"
    )

    lr_tuned = evaluate_model(
        best_lr,
        X_test_tfidf,
        y_test,
        "Logistic Regression - Tuned"
    )

    # ========================================================
    # 2. MULTINOMIAL NAIVE BAYES - BASELINE
    # ========================================================

    print("\n" + "=" * 70)
    print("2. MULTINOMIAL NAIVE BAYES")
    print("=" * 70)

    nb_model = MultinomialNB()

    nb_model.fit(
        X_train_tfidf,
        y_train
    )

    nb_baseline = evaluate_model(
        nb_model,
        X_test_tfidf,
        y_test,
        "Multinomial Naive Bayes - Baseline"
    )

    # ========================================================
    # MULTINOMIAL NAIVE BAYES - GRID SEARCH
    # ========================================================

    nb_param_grid = {
        "alpha": [
            0.01,
            0.1,
            0.5,
            1,
            2,
            5,
            10
        ]
    }

    nb_grid = GridSearchCV(
        estimator=MultinomialNB(),
        param_grid=nb_param_grid,
        scoring=f1_scorer,
        cv=5,
        n_jobs=1
    )

    nb_grid.fit(
        X_train_tfidf,
        y_train
    )

    best_nb = nb_grid.best_estimator_

    print("\nBest Naive Bayes parameters:")
    print(
        nb_grid.best_params_
    )

    print(
        f"Best Naive Bayes CV F1: "
        f"{nb_grid.best_score_:.4f}"
    )

    nb_tuned = evaluate_model(
        best_nb,
        X_test_tfidf,
        y_test,
        "Multinomial Naive Bayes - Tuned"
    )

    # ========================================================
    # 3. LINEAR SVC - BASELINE
    # ========================================================

    print("\n" + "=" * 70)
    print("3. LINEAR SVC")
    print("=" * 70)

    svm_model = LinearSVC(
        class_weight="balanced",
        random_state=42
    )

    svm_model.fit(
        X_train_tfidf,
        y_train
    )

    svm_baseline = evaluate_model(
        svm_model,
        X_test_tfidf,
        y_test,
        "Linear SVC - Baseline"
    )

    # ========================================================
    # LINEAR SVC - GRID SEARCH
    # ========================================================

    svm_param_grid = {
        "C": [
            0.01,
            0.1,
            1,
            10,
            100
        ]
    }

    svm_grid = GridSearchCV(
        estimator=LinearSVC(
            class_weight="balanced",
            random_state=42
        ),
        param_grid=svm_param_grid,
        scoring=f1_scorer,
        cv=5,
        n_jobs=1
    )

    svm_grid.fit(
        X_train_tfidf,
        y_train
    )

    best_svm = svm_grid.best_estimator_

    print("\nBest Linear SVC parameters:")
    print(
        svm_grid.best_params_
    )

    print(
        f"Best Linear SVC CV F1: "
        f"{svm_grid.best_score_:.4f}"
    )

    svm_tuned = evaluate_model(
        best_svm,
        X_test_tfidf,
        y_test,
        "Linear SVC - Tuned"
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    comparison = pd.DataFrame([
        lr_tuned,
        nb_tuned,
        svm_tuned
    ])

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison
    )

    # ========================================================
    # COMPARISON CHART
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    comparison_plot = comparison.set_index(
        "Model"
    )[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ]

    comparison_plot.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title(
        "Comparison of Tuned Sentiment Models"
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Score"
    )

    plt.xticks(
        rotation=0
    )

    plt.ylim(
        0,
        1
    )

    plt.legend(
        title="Metric"
    )

    plt.tight_layout()

    plt.show()

    # ========================================================
    # SELECT FINAL MODEL
    #
    # ORIGINAL NOTEBOOK:
    # LOGISTIC REGRESSION IS THE FINAL MODEL
    # ========================================================

    best_model_name = "Logistic Regression - Tuned"

    best_model = best_lr

    print(
        "\nBest model based on notebook selection:"
    )

    print(
        best_model_name
    )

    print(
        "\nFinal model object:"
    )

    print(
        type(best_model).__name__
    )

    # ========================================================
    # FINAL MODEL METRICS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL SELECTED MODEL METRICS")
    print("=" * 70)

    print(
        f"Accuracy : {lr_tuned['Accuracy']:.4f}"
    )

    print(
        f"Precision: {lr_tuned['Precision']:.4f}"
    )

    print(
        f"Recall   : {lr_tuned['Recall']:.4f}"
    )

    print(
        f"F1 Score : {lr_tuned['F1 Score']:.4f}"
    )

    # ========================================================
    # RETURN ALL IMPORTANT ARTIFACTS
    # ========================================================

    return {

        # ----------------------------------------------------
        # TF-IDF
        # ----------------------------------------------------

        "tfidf": tfidf,

        # ----------------------------------------------------
        # RAW TRAIN / TEST DATA
        # ----------------------------------------------------

        "X_train": X_train,
        "X_test": X_test,

        "y_train": y_train,
        "y_test": y_test,

        # ----------------------------------------------------
        # TF-IDF TRAIN / TEST DATA
        # ----------------------------------------------------

        "X_train_tfidf": X_train_tfidf,
        "X_test_tfidf": X_test_tfidf,

        # ----------------------------------------------------
        # LOGISTIC REGRESSION
        # ----------------------------------------------------

        "lr_model": lr_model,
        "best_lr": best_lr,
        "lr_grid": lr_grid,

        # ----------------------------------------------------
        # NAIVE BAYES
        # ----------------------------------------------------

        "nb_model": nb_model,
        "best_nb": best_nb,
        "nb_grid": nb_grid,

        # ----------------------------------------------------
        # LINEAR SVC
        # ----------------------------------------------------

        "svm_model": svm_model,
        "best_svm": best_svm,
        "svm_grid": svm_grid,

        # ----------------------------------------------------
        # MODEL RESULTS
        # ----------------------------------------------------

        "lr_baseline": lr_baseline,
        "lr_tuned": lr_tuned,

        "nb_baseline": nb_baseline,
        "nb_tuned": nb_tuned,

        "svm_baseline": svm_baseline,
        "svm_tuned": svm_tuned,

        # ----------------------------------------------------
        # COMPARISON
        # ----------------------------------------------------

        "comparison": comparison,

        # ----------------------------------------------------
        # FINAL MODEL
        # ----------------------------------------------------

        "best_model": best_model,

        "best_model_name": best_model_name
    }