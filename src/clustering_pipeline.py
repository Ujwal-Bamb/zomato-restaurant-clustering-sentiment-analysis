# ============================================================
# ZOMATO RESTAURANT CLUSTERING & SENTIMENT ANALYSIS
# CLUSTERING PIPELINE
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

from scipy.cluster.hierarchy import (
    dendrogram,
    linkage
)


# ============================================================
# CLUSTERING FUNCTION
# ============================================================

def run_clustering(feature_results, review):

    print("\n" + "=" * 70)
    print("CLUSTERING")
    print("=" * 70)


    # ========================================================
    # GET FEATURE DATA
    # ========================================================

    restaurant = feature_results["restaurant"].copy()

    restaurant_pca = feature_results["restaurant_pca"]


    # ========================================================
    # CREATE AVG_RATING
    # ========================================================
    #
    # Avg_Rating is required later for cluster profiles.
    #
    # We calculate the average rating for every restaurant
    # from the review dataset and map it to restaurant Name.
    #
    # ========================================================

    avg_rating = (
        review
        .groupby("Restaurant")["Rating"]
        .mean()
    )

    restaurant["Avg_Rating"] = (
        restaurant["Name"].map(avg_rating)
    )


    # ========================================================
    # CHECK AVG_RATING
    # ========================================================

    print("\nChecking Avg_Rating column...")

    print(
        "Avg_Rating exists:",
        "Avg_Rating" in restaurant.columns
    )

    print(
        restaurant[
            [
                "Name",
                "Avg_Rating"
            ]
        ].head()
    )


    # ========================================================
    # K-MEANS CLUSTERING
    # TEST K = 2 TO 10
    # ========================================================

    k_values = range(2, 11)

    inertia_values = []

    silhouette_values = []


    for k in k_values:

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(
            restaurant_pca
        )

        inertia_values.append(
            kmeans.inertia_
        )

        silhouette_values.append(
            silhouette_score(
                restaurant_pca,
                labels
            )
        )


    # ========================================================
    # ELBOW CURVE
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        k_values,
        inertia_values,
        marker="o"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "Inertia"
    )

    plt.title(
        "K-Means Elbow Method"
    )

    plt.grid(True)

    plt.show()


    # ========================================================
    # SILHOUETTE SCORE
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        k_values,
        silhouette_values,
        marker="o"
    )

    plt.xlabel(
        "Number of Clusters (K)"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.title(
        "Silhouette Score for K-Means"
    )

    plt.grid(True)

    plt.show()


    # ========================================================
    # FINAL K-MEANS
    # K = 9
    # ========================================================

    final_kmeans = KMeans(
        n_clusters=9,
        random_state=42,
        n_init=10
    )

    kmeans_labels = final_kmeans.fit_predict(
        restaurant_pca
    )

    restaurant["KMeans_Cluster"] = (
        kmeans_labels
    )


    # ========================================================
    # K-MEANS EVALUATION
    # ========================================================

    kmeans_silhouette = silhouette_score(
        restaurant_pca,
        kmeans_labels
    )

    kmeans_calinski = calinski_harabasz_score(
        restaurant_pca,
        kmeans_labels
    )

    kmeans_davies = davies_bouldin_score(
        restaurant_pca,
        kmeans_labels
    )


    print("\nK-Means Results")

    print("-" * 40)

    print(
        f"Silhouette Score: "
        f"{kmeans_silhouette:.4f}"
    )

    print(
        f"Calinski-Harabasz Score: "
        f"{kmeans_calinski:.4f}"
    )

    print(
        f"Davies-Bouldin Score: "
        f"{kmeans_davies:.4f}"
    )


    # ========================================================
    # K-MEANS CLUSTER VISUALIZATION
    # ========================================================

    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(
        restaurant_pca[:, 0],
        restaurant_pca[:, 1],
        c=kmeans_labels,
        cmap="tab10",
        s=70,
        alpha=0.8
    )

    plt.xlabel(
        "Principal Component 1"
    )

    plt.ylabel(
        "Principal Component 2"
    )

    plt.title(
        "K-Means Clusters"
    )

    plt.colorbar(
        scatter,
        label="Cluster"
    )

    plt.show()


    # ========================================================
    # HIERARCHICAL CLUSTERING
    # DENDROGRAM
    # ========================================================

    linked = linkage(
        restaurant_pca,
        method="ward"
    )

    plt.figure(figsize=(12, 6))

    dendrogram(
        linked,
        truncate_mode="lastp",
        p=20
    )

    plt.title(
        "Hierarchical Clustering Dendrogram"
    )

    plt.xlabel(
        "Cluster"
    )

    plt.ylabel(
        "Distance"
    )

    plt.show()


    # ========================================================
    # AGGLOMERATIVE CLUSTERING
    # TEST K = 2 TO 10
    # ========================================================

    hierarchical_k_values = range(2, 11)

    hierarchical_silhouette = []


    for k in hierarchical_k_values:

        hierarchical_model = (
            AgglomerativeClustering(
                n_clusters=k
            )
        )

        hierarchical_labels = (
            hierarchical_model.fit_predict(
                restaurant_pca
            )
        )

        hierarchical_silhouette.append(
            silhouette_score(
                restaurant_pca,
                hierarchical_labels
            )
        )


    # ========================================================
    # HIERARCHICAL SILHOUETTE SCORE
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        hierarchical_k_values,
        hierarchical_silhouette,
        marker="o"
    )

    plt.xlabel(
        "Number of Clusters"
    )

    plt.ylabel(
        "Silhouette Score"
    )

    plt.title(
        "Silhouette Score for Agglomerative Clustering"
    )

    plt.grid(True)

    plt.show()


    # ========================================================
    # FINAL AGGLOMERATIVE CLUSTERING
    # K = 2
    # ========================================================

    final_hierarchical = (
        AgglomerativeClustering(
            n_clusters=2
        )
    )

    hierarchical_labels = (
        final_hierarchical.fit_predict(
            restaurant_pca
        )
    )

    restaurant["Hierarchical_Cluster"] = (
        hierarchical_labels
    )


    # ========================================================
    # HIERARCHICAL EVALUATION
    # ========================================================

    hierarchical_silhouette_score = (
        silhouette_score(
            restaurant_pca,
            hierarchical_labels
        )
    )

    hierarchical_calinski = (
        calinski_harabasz_score(
            restaurant_pca,
            hierarchical_labels
        )
    )

    hierarchical_davies = (
        davies_bouldin_score(
            restaurant_pca,
            hierarchical_labels
        )
    )


    print(
        "\nAgglomerative Clustering Results"
    )

    print("-" * 40)

    print(
        f"Silhouette Score: "
        f"{hierarchical_silhouette_score:.4f}"
    )

    print(
        f"Calinski-Harabasz Score: "
        f"{hierarchical_calinski:.4f}"
    )

    print(
        f"Davies-Bouldin Score: "
        f"{hierarchical_davies:.4f}"
    )


    # ========================================================
    # HIERARCHICAL CLUSTER VISUALIZATION
    # ========================================================

    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(
        restaurant_pca[:, 0],
        restaurant_pca[:, 1],
        c=hierarchical_labels,
        cmap="viridis",
        s=70,
        alpha=0.8
    )

    plt.xlabel(
        "Principal Component 1"
    )

    plt.ylabel(
        "Principal Component 2"
    )

    plt.title(
        "Agglomerative Clustering (K=2)"
    )

    plt.colorbar(
        scatter,
        label="Cluster"
    )

    plt.show()


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    clustering_comparison = pd.DataFrame({

        "Model": [
            "K-Means",
            "Agglomerative Clustering"
        ],

        "Clusters": [
            9,
            2
        ],

        "Silhouette Score": [
            kmeans_silhouette,
            hierarchical_silhouette_score
        ],

        "Calinski-Harabasz Score": [
            kmeans_calinski,
            hierarchical_calinski
        ],

        "Davies-Bouldin Score": [
            kmeans_davies,
            hierarchical_davies
        ]

    })


    print(
        "\nClustering Model Comparison"
    )

    print(
        clustering_comparison
    )


    # ========================================================
    # K-MEANS CLUSTER PROFILES
    # ========================================================

    kmeans_profile = (

        restaurant

        .groupby(
            "KMeans_Cluster"
        )

        .agg(

            Restaurant_Count=(
                "Name",
                "count"
            ),

            Avg_Cost=(
                "Cost",
                "mean"
            ),

            Avg_Rating=(
                "Avg_Rating",
                "mean"
            ),

            Avg_Cuisine_Count=(
                "Cuisine_Count",
                "mean"
            )

        )

        .reset_index()

    )


    print(
        "\nK-Means Cluster Profiles"
    )

    print(
        kmeans_profile
    )


    # ========================================================
    # HIERARCHICAL CLUSTER PROFILES
    # ========================================================

    hierarchical_profile = (

        restaurant

        .groupby(
            "Hierarchical_Cluster"
        )

        .agg(

            Restaurant_Count=(
                "Name",
                "count"
            ),

            Avg_Cost=(
                "Cost",
                "mean"
            ),

            Avg_Rating=(
                "Avg_Rating",
                "mean"
            ),

            Avg_Cuisine_Count=(
                "Cuisine_Count",
                "mean"
            )

        )

        .reset_index()

    )


    print(
        "\nHierarchical Cluster Profiles"
    )

    print(
        hierarchical_profile
    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "restaurant": restaurant,

        "kmeans_model": final_kmeans,

        "hierarchical_model": final_hierarchical,

        "kmeans_labels": kmeans_labels,

        "hierarchical_labels": hierarchical_labels,

        "kmeans_profile": kmeans_profile,

        "hierarchical_profile": hierarchical_profile,

        "comparison": clustering_comparison

    }