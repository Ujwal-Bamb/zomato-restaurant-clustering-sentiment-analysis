# ============================================================
# ZOMATO RESTAURANT CLUSTERING & SENTIMENT ANALYSIS
# SECTION 1: DATA LOADING
# ============================================================

import pandas as pd


def load_data(
    restaurant_path="data/Zomato Restaurant names and Metadata (1).csv",
    review_path="data/Zomato Restaurant reviews (1).csv",
):
    """
    Load both datasets once and return them.
    """

    restaurant_df = pd.read_csv(restaurant_path)
    review_df = pd.read_csv(review_path)

    print("Restaurant Dataset Shape:", restaurant_df.shape)
    print("Review Dataset Shape:", review_df.shape)

    return restaurant_df, review_df


if __name__ == "__main__":
    restaurant_df, review_df = load_data()