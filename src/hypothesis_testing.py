# ============================================================
# ZOMATO RESTAURANT CLUSTERING & SENTIMENT ANALYSIS
# HYPOTHESIS TESTING
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


def run_hypothesis_tests(restaurant, review):

    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTING")
    print("=" * 70)


    # ========================================================
    # TEST 1
    # COST VS AVERAGE RESTAURANT RATING
    # ========================================================

    print("\n--- Test 1: Cost vs Average Restaurant Rating ---")

    restaurant_rating = (
        review
        .groupby('Restaurant')['Rating']
        .mean()
        .reset_index(name='Avg_Rating')
    )

    rating_cost = restaurant_rating.merge(
        restaurant[['Name', 'Cost']],
        left_on='Restaurant',
        right_on='Name',
        how='inner'
    )

    correlation, p_value = stats.spearmanr(
        rating_cost['Cost'],
        rating_cost['Avg_Rating']
    )

    print(f"Spearman Correlation: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")

    alpha = 0.05

    if p_value < alpha:
        print("Reject H0: There is a significant relationship between Cost and Average Rating.")
    else:
        print("Fail to Reject H0: There is no significant relationship between Cost and Average Rating.")


    # ========================================================
    # TEST 2
    # PROPORTION OF HIGH-RATED REVIEWS
    # ========================================================

    print("\n--- Test 2: Proportion of High-Rated Reviews ---")

    rated_reviews = review['Rating'].dropna()

    high_rated = (rated_reviews >= 4).sum()

    total_rated = len(rated_reviews)

    observed_proportion = high_rated / total_rated

    print(f"Total rated reviews: {total_rated}")
    print(f"High-rated reviews: {high_rated}")
    print(
        f"Observed high-rated proportion: "
        f"{observed_proportion:.4f}"
    )

    # H0: p = 0.50
    # H1: p > 0.50

    count = high_rated
    nobs = total_rated

    z_stat, p_value = proportions_ztest(
        count=count,
        nobs=nobs,
        value=0.50,
        alternative='larger'
    )

    print(f"Z-statistic: {z_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < alpha:
        print(
            "Reject H0: More than 50% of the reviews "
            "are high-rated."
        )
    else:
        print(
            "Fail to Reject H0: There is not enough evidence "
            "that more than 50% of reviews are high-rated."
        )


    # ========================================================
    # TEST 3
    # CUISINE COUNT VS COST
    # ========================================================

    print("\n--- Test 3: Cuisine Count vs Cost ---")

    restaurant_test = restaurant.copy()

    restaurant_test['Cuisine_Count'] = (
        restaurant_test['Cuisines']
        .apply(
            lambda x: len(
                [c.strip() for c in x.split(',')]
            )
        )
    )

    correlation, p_value = stats.spearmanr(
        restaurant_test['Cuisine_Count'],
        restaurant_test['Cost']
    )

    print(
        f"Spearman Correlation: "
        f"{correlation:.4f}"
    )

    print(
        f"P-value: "
        f"{p_value:.8f}"
    )

    if p_value < alpha:
        print(
            "Reject H0: There is a significant relationship "
            "between Cuisine Count and Cost."
        )
    else:
        print(
            "Fail to Reject H0: There is no significant relationship "
            "between Cuisine Count and Cost."
        )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    results = {
        "cost_vs_rating": {
            "spearman_correlation": correlation,
            "p_value": p_value
        },
        "high_rating_proportion": {
            "observed_proportion": observed_proportion,
            "z_statistic": z_stat,
            "p_value": p_value
        },
        "cuisine_count_vs_cost": {
            "spearman_correlation": correlation,
            "p_value": p_value
        }
    }

    return results