# ============================================================
# ZOMATO RESTAURANT CLUSTERING & SENTIMENT ANALYSIS
# FEATURE ENGINEERING
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import PCA


def run_feature_engineering(restaurant, review):

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    restaurant = restaurant.copy()

    # ========================================================
    # AVERAGE RESTAURANT RATING
    # ========================================================

    avg_rating = (
        review
        .groupby('Restaurant')['Rating']
        .mean()
        .reset_index(name='Avg_Rating')
    )

    restaurant = restaurant.merge(
        avg_rating,
        left_on='Name',
        right_on='Restaurant',
        how='left'
    )

    # ========================================================
    # CUISINE COUNT
    # ========================================================

    restaurant['Cuisine_Count'] = (
        restaurant['Cuisines']
        .apply(
            lambda x: len(
                [c.strip() for c in x.split(',')]
            )
        )
    )

    print("\nCuisine Count:")
    print(
        restaurant[
            ['Name', 'Cuisines', 'Cuisine_Count']
        ].head()
    )

    # ========================================================
    # MULTI-LABEL BINARIZATION OF CUISINES
    # ========================================================

    cuisine_lists = restaurant['Cuisines'].apply(
        lambda x: [c.strip() for c in x.split(',')]
    )

    mlb = MultiLabelBinarizer()

    cuisine_encoded = mlb.fit_transform(
        cuisine_lists
    )

    cuisine_encoded_df = pd.DataFrame(
        cuisine_encoded,
        columns=mlb.classes_,
        index=restaurant.index
    )

    print("\nNumber of cuisine features:")
    print(len(mlb.classes_))

    # ========================================================
    # CREATE FEATURE MATRIX
    # ========================================================

    features = pd.concat(
        [
            restaurant[
                [
                    'Cost',
                    'Cuisine_Count'
                ]
            ],
            cuisine_encoded_df
        ],
        axis=1
    )

    print("\nFeature matrix shape:")
    print(features.shape)

    # ========================================================
    # LOG TRANSFORMATION OF COST
    # ========================================================

    restaurant['Log_Cost'] = np.log1p(
        restaurant['Cost']
    )

    # Original Cost vs Log Cost

    plt.figure(figsize=(10, 5))

    plt.hist(
        restaurant['Cost'],
        bins=20,
        alpha=0.7
    )

    plt.title('Original Cost Distribution')
    plt.xlabel('Cost')
    plt.ylabel('Frequency')

    plt.show()


    plt.figure(figsize=(10, 5))

    plt.hist(
        restaurant['Log_Cost'],
        bins=20,
        alpha=0.7
    )

    plt.title('Log-Transformed Cost Distribution')
    plt.xlabel('Log Cost')
    plt.ylabel('Frequency')

    plt.show()

    # ========================================================
    # FINAL FEATURES FOR CLUSTERING
    # ========================================================

    final_features = pd.concat(
        [
            restaurant[
                [
                    'Log_Cost',
                    'Cuisine_Count'
                ]
            ],
            cuisine_encoded_df
        ],
        axis=1
    )

    print("\nFinal feature matrix:")
    print(final_features.head())

    print("\nFinal feature matrix shape:")
    print(final_features.shape)

    # ========================================================
    # STANDARDIZATION
    # ========================================================

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        final_features
    )

    scaled_features_df = pd.DataFrame(
        scaled_features,
        columns=final_features.columns,
        index=final_features.index
    )

    print("\nScaled feature matrix:")
    print(scaled_features_df.head())

    # ========================================================
    # PCA
    # RETAIN 90% OF VARIANCE
    # ========================================================

    pca = PCA(
        n_components=0.90
    )

    restaurant_pca = pca.fit_transform(
        scaled_features
    )

    print("\nNumber of PCA components:")
    print(pca.n_components_)

    print("\nExplained variance ratio:")
    print(pca.explained_variance_ratio_)

    print("\nTotal explained variance:")
    print(
        pca.explained_variance_ratio_.sum()
    )

    # ========================================================
    # CUMULATIVE EXPLAINED VARIANCE
    # ========================================================

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        range(
            1,
            len(cumulative_variance) + 1
        ),
        cumulative_variance,
        marker='o'
    )

    plt.axhline(
        y=0.90,
        linestyle='--'
    )

    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')

    plt.grid(True)

    plt.show()

    # ========================================================
    # RETURN EVERYTHING REQUIRED BY CLUSTERING
    # ========================================================

    return {
        'restaurant': restaurant,
        'features': final_features,
        'scaled_features': scaled_features,
        'scaled_features_df': scaled_features_df,
        'cuisine_encoder': mlb,
        'scaler': scaler,
        'pca': pca,
        'restaurant_pca': restaurant_pca
    }