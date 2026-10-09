# ============================================================
# ZOMATO RESTAURANT CLUSTERING & SENTIMENT ANALYSIS
# NLP PREPROCESSING
# ============================================================

import re
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from contractions import fix


# ============================================================
# DOWNLOAD NLTK RESOURCES
# ============================================================

def download_nltk_resources():

    resources = [
        "stopwords",
        "punkt",
        "punkt_tab",
        "wordnet",
        "omw-1.4"
    ]

    for resource in resources:

        try:

            nltk.download(
                resource,
                quiet=True
            )

        except Exception:

            pass


# ============================================================
# CLEAN REVIEW TEXT
# ============================================================

def clean_review_text(
    text,
    stop_words,
    lemmatizer
):

    if pd.isna(text):

        return ""

    # --------------------------------------------------------
    # Expand contractions
    # --------------------------------------------------------

    text = fix(
        str(text)
    )

    # --------------------------------------------------------
    # Lowercase
    # --------------------------------------------------------

    text = text.lower()

    # --------------------------------------------------------
    # Remove URLs
    # --------------------------------------------------------

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # --------------------------------------------------------
    # Remove tokens containing digits
    # --------------------------------------------------------

    text = re.sub(
        r"\S*\d\S*",
        " ",
        text
    )

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    tokens = word_tokenize(
        text
    )

    # --------------------------------------------------------
    # Keep alphabetic tokens
    # --------------------------------------------------------

    tokens = [
        token
        for token in tokens
        if token.isalpha()
    ]

    # --------------------------------------------------------
    # Remove stopwords
    # Preserve negations
    # --------------------------------------------------------

    tokens = [
        token
        for token in tokens
        if (
            token not in stop_words
            or token in {
                "no",
                "not",
                "nor",
                "never"
            }
        )
    ]

    # --------------------------------------------------------
    # Lemmatization
    # --------------------------------------------------------

    tokens = [
        lemmatizer.lemmatize(
            token
        )
        for token in tokens
    ]

    # --------------------------------------------------------
    # Join tokens
    # --------------------------------------------------------

    return " ".join(
        tokens
    )


# ============================================================
# MAIN NLP FUNCTION
# ============================================================

def run_nlp_preprocessing(
    review
):

    print("\n" + "=" * 70)
    print("NLP PREPROCESSING")
    print("=" * 70)

    # ========================================================
    # DOWNLOAD RESOURCES
    # ========================================================

    download_nltk_resources()

    # ========================================================
    # COPY DATA
    # ========================================================

    review = review.copy()

    # ========================================================
    # KEEP REVIEWS WITH TEXT AND RATING
    #
    # MATCHES ORIGINAL NOTEBOOK
    # ========================================================

    review_nlp = review[
        review["Review"].notna()
        &
        review["Rating"].notna()
    ].copy()

    print(
        "\nReviews available for NLP:",
        len(review_nlp)
    )

    # ========================================================
    # ENSURE RATING IS NUMERIC
    # ========================================================

    review_nlp["Rating"] = pd.to_numeric(
        review_nlp["Rating"],
        errors="coerce"
    )

    # Remove any rows where Rating became missing
    review_nlp = review_nlp[
        review_nlp["Rating"].notna()
    ].copy()

    # ========================================================
    # CREATE SENTIMENT TARGET
    #
    # IMPORTANT:
    # CREATE THE COLUMN DIRECTLY.
    #
    # This avoids the ArrowStringArray assignment error.
    # ========================================================

    review_nlp["Sentiment"] = review_nlp[
        "Rating"
    ].apply(
        lambda x:
        "Positive"
        if x >= 4
        else "Negative"
    )

    # ========================================================
    # SENTIMENT DISTRIBUTION
    # ========================================================

    print(
        "\nSentiment Distribution:"
    )

    print(
        review_nlp[
            "Sentiment"
        ].value_counts()
    )

    print(
        "\nSentiment Percentage:"
    )

    print(
        (
            review_nlp[
                "Sentiment"
            ]
            .value_counts(
                normalize=True
            )
            * 100
        ).round(2)
    )

    # ========================================================
    # SAMPLE SENTIMENT LABELS
    # ========================================================

    print(
        "\nSample Sentiment Labels:"
    )

    print(
        review_nlp[
            [
                "Rating",
                "Sentiment"
            ]
        ].head(10)
    )

    # ========================================================
    # STOPWORDS
    # ========================================================

    stop_words = set(
        stopwords.words(
            "english"
        )
    )

    # Preserve important negations

    for word in [
        "no",
        "not",
        "nor",
        "never"
    ]:

        stop_words.discard(
            word
        )

    # ========================================================
    # LEMMATIZER
    # ========================================================

    lemmatizer = WordNetLemmatizer()

    # ========================================================
    # PROCESS REVIEW TEXT
    # ========================================================

    print(
        "\nProcessing review text..."
    )

    review_nlp[
        "Processed_Review"
    ] = review_nlp[
        "Review"
    ].apply(
        lambda text:
        clean_review_text(
            text,
            stop_words,
            lemmatizer
        )
    )

    # ========================================================
    # REMOVE EMPTY PROCESSED REVIEWS
    # ========================================================

    review_nlp = review_nlp[
        review_nlp[
            "Processed_Review"
        ].str.strip().ne("")
    ].copy()

    # ========================================================
    # PRINT SAMPLE PROCESSED REVIEWS
    # ========================================================

    print(
        "\nSample Processed Reviews:"
    )

    print(
        review_nlp[
            [
                "Review",
                "Processed_Review",
                "Sentiment"
            ]
        ].head(5)
    )

    # ========================================================
    # PREPARE X AND y
    # ========================================================

    X = review_nlp[
        "Processed_Review"
    ]

    y = review_nlp[
        "Sentiment"
    ]

    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print(
        "\nFinal NLP dataset shape:"
    )

    print(
        review_nlp.shape
    )

    print(
        "\nFinal sentiment distribution:"
    )

    print(
        y.value_counts()
    )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "review": review_nlp,

        "X": X,

        "y": y,

        "stop_words": stop_words,

        "lemmatizer": lemmatizer
    }