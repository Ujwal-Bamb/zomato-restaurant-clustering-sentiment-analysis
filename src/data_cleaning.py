# ============================================================
# ZOMATO RESTAURANT CLUSTERING & SENTIMENT ANALYSIS
# SECTION 2: DATA UNDERSTANDING & DATA WRANGLING
# ============================================================

import pandas as pd


def inspect_data(restaurant_df, review_df):

    print("\nTop 5 Rows of Restaurant Dataset")
    print(restaurant_df.head())

    print("\nTop 5 Rows of Review Dataset")
    print(review_df.head())

    print("\nRestaurant Dataset:")
    print(
        f"Rows: {restaurant_df.shape[0]} "
        f"and Columns: {restaurant_df.shape[1]}"
    )

    print("\nReview Dataset:")
    print(
        f"Rows: {review_df.shape[0]} "
        f"and Columns: {review_df.shape[1]}"
    )

    print("\nRestaurant Dataset Information:")
    restaurant_df.info()

    print("\n" + "=" * 60)

    print("\nReview Dataset Information:")
    review_df.info()

    print("\nRestaurant Dataset Duplicate Values:")
    print(
        f"Total Duplicate Values: "
        f"{restaurant_df.duplicated().sum()}"
    )

    print("\nReview Dataset Duplicate Values:")
    print(
        f"Total Duplicate Values: "
        f"{review_df.duplicated().sum()}"
    )

    print("\nRestaurant Dataset Missing Values:")
    print(restaurant_df.isnull().sum())

    print("\nReview Dataset Missing Values:")
    print(review_df.isnull().sum())

    print("\nRestaurant Dataset Columns:")
    print(restaurant_df.columns)

    print("\nReview Dataset Columns:")
    print(review_df.columns)

    print("\nRestaurant Dataset Describe:")
    print(restaurant_df.describe(include="all"))

    print("\nReview Dataset Describe:")
    print(review_df.describe(include="all"))

    print("\nUnique Values in Restaurant Dataset:")
    print(restaurant_df.nunique())

    print("\nList of unique Cost:")
    print(restaurant_df["Cost"].unique())

    print("\nList of unique Collections:")
    print(restaurant_df["Collections"].unique())

    print("\nUnique Values in Review Dataset:")
    print(review_df.nunique())

    print("\nList of unique Ratings:")
    print(review_df["Rating"].unique())

    print("\nList of unique Pictures:")
    print(review_df["Pictures"].unique())


def clean_data(restaurant_df, review_df):

    print("\nRestaurant Dataset Duplicates:")
    print(restaurant_df.duplicated().sum())

    print("\nReview Dataset Duplicates:")
    print(review_df.duplicated().sum())

    duplicate_reviews = review_df[
        review_df.duplicated(keep=False)
    ]

    print("\nDuplicate review records:")
    print(duplicate_reviews.head(10))

    # --------------------------------------------------------
    # Create working copies
    # --------------------------------------------------------

    restaurant = restaurant_df.copy()
    review = review_df.copy()

    # --------------------------------------------------------
    # Remove duplicate review rows
    # --------------------------------------------------------

    review = review.drop_duplicates()

    print(
        "\nShape after removing duplicates:",
        review.shape
    )

    print(
        "Remaining duplicates:",
        review.duplicated().sum()
    )

    # --------------------------------------------------------
    # Restaurant missing values
    # --------------------------------------------------------

    print("\nRestaurant Dataset:")
    print(restaurant.isnull().sum())

    print("\nMissing Collections:")
    print(
        restaurant[
            restaurant["Collections"].isnull()
        ]
    )

    print("\nMissing Timings:")
    print(
        restaurant[
            restaurant["Timings"].isnull()
        ]
    )

    # Same treatment as notebook
    restaurant["Collections"] = (
        restaurant["Collections"]
        .fillna("No Collection")
    )

    restaurant["Timings"] = (
        restaurant["Timings"]
        .fillna("Unknown")
    )

    print(
        "\nRestaurant missing values after treatment:"
    )

    print(restaurant.isnull().sum())

    # --------------------------------------------------------
    # Review missing values
    # --------------------------------------------------------

    print("\nReview Dataset:")
    print(review.isnull().sum())

    missing_review_rows = review[
        review.isnull().any(axis=1)
    ]

    print(
        "\nRows with missing values:",
        missing_review_rows.shape[0]
    )

    print(
        missing_review_rows.head(20)
    )

    print(
        "\nMissing-value count per row:"
    )

    print(
        review.isnull()
        .sum(axis=1)
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Remove rows where all important review fields are missing
    # --------------------------------------------------------

    review = review.dropna(
        subset=[
            "Reviewer",
            "Review",
            "Rating",
            "Metadata",
            "Time",
        ],
        how="all",
    )

    print(
        "\nShape after removing incomplete rows:",
        review.shape
    )

    print("\nMissing values:")
    print(review.isnull().sum())

    # --------------------------------------------------------
    # Data types before conversion
    # --------------------------------------------------------

    print("\nCurrent Data Types")

    print("Restaurant Data Types:")
    print(restaurant.dtypes)

    print("\nReview Data Types:")
    print(review.dtypes)

    # --------------------------------------------------------
    # Data type conversion
    # --------------------------------------------------------

    restaurant["Cost"] = (
        restaurant["Cost"]
        .str.replace(",", "", regex=False)
        .astype(int)
    )

    review["Rating"] = pd.to_numeric(
        review["Rating"],
        errors="coerce",
    )

    review["Time"] = pd.to_datetime(
        review["Time"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Data types after conversion
    # --------------------------------------------------------

    print("\nData Types After Conversion")

    print("Restaurant Data Types:")
    print(restaurant.dtypes)

    print("\nReview Data Types:")
    print(review.dtypes)

    print("\nMissing Rating values:")
    print(review["Rating"].isnull().sum())

    print("\nRating distribution:")
    print(
        review["Rating"]
        .value_counts(dropna=False)
        .sort_index()
    )

    # --------------------------------------------------------
    # Final data quality check
    # --------------------------------------------------------

    print("\nFinal data quality check")

    print(
        "Restaurant shape:",
        restaurant.shape
    )

    print(
        "Review shape:",
        review.shape
    )

    print("\nRestaurant missing values:")
    print(restaurant.isnull().sum())

    print("\nReview missing values:")
    print(review.isnull().sum())

    print("\nReview data types:")
    print(review.dtypes)

    return restaurant, review