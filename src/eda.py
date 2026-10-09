# ============================================================
# ZOMATO RESTAURANT CLUSTERING & SENTIMENT ANALYSIS
# EDA - EXPLORATORY DATA ANALYSIS
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(restaurant, review):

    # ========================================================
    # CHART 1 - COST DISTRIBUTION - HISTOGRAM
    # ========================================================

    plt.figure(figsize=(10, 5))

    sns.histplot(
        restaurant['Cost'],
        bins=20,
        kde=True,
        edgecolor='white',
        color='teal'
    )

    plt.title('Distribution of Restaurant Costs')
    plt.xlabel('Cost')
    plt.ylabel('Number of Restaurants')

    plt.show()

    print(f"Skewness: {restaurant['Cost'].skew():.4f}")
    print(
        f"Mean  : {restaurant['Cost'].mean():.0f} "
        f"Median: {restaurant['Cost'].median():.0f}"
    )


    # ========================================================
    # CHART 2 - PICTURES PER REVIEW DISTRIBUTION
    # ========================================================

    top_15 = review['Pictures'].value_counts().head(15)

    plt.figure(figsize=(10, 5))

    ax = sns.barplot(
        x=top_15.index,
        y=top_15.values,
        color='steelblue',
        palette='viridis'
    )

    plt.title('Top 15 Pictures per Review')
    plt.xlabel('Number of Pictures')
    plt.ylabel('Number of Reviews')

    plt.show()


    # ========================================================
    # CHART 3 - CUISINE DISTRIBUTION
    # ========================================================

    cuisine_list = []

    for row in restaurant['Cuisines']:
        for cuisine in row.split(','):
            cuisine_list.append(cuisine.strip())

    print("Total cuisine entries:", len(cuisine_list))
    print(cuisine_list[:20])


    from collections import Counter

    cuisine_counts = Counter(cuisine_list)

    cuisine_df = pd.DataFrame(
        cuisine_counts.most_common(15),
        columns=['Cuisine', 'Count']
    )

    print(cuisine_df)


    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=cuisine_df,
        x='Count',
        y='Cuisine',
        palette='mako'
    )

    plt.xlabel('Number of Restaurants')
    plt.ylabel('Cuisine')
    plt.title('Top 15 Most Common Cuisines')

    plt.show()


    # ========================================================
    # CHART 4 - NUMBER OF CUISINES PER RESTAURANT
    # ========================================================

    restaurant['Cuisine_Count'] = restaurant['Cuisines'].apply(
        lambda x: len(x.split(','))
    )

    print(
        restaurant[
            ['Name', 'Cuisine_Count']
        ].sort_values(
            by='Cuisine_Count',
            ascending=False
        ).head(10)
    )

    print(restaurant['Cuisine_Count'].describe())


    plt.figure(figsize=(10, 5))

    sns.histplot(
        restaurant['Cuisine_Count'],
        bins=10,
        kde=True,
        color='steelblue'
    )

    plt.title('Distribution of Number of Cuisines per Restaurant')
    plt.xlabel('Number of Cuisines')
    plt.ylabel('Number of Restaurants')

    plt.show()


    # ========================================================
    # CHART 5 - RATING DISTRIBUTION
    # ========================================================

    plt.figure(figsize=(10, 5))

    sns.countplot(
        data=review,
        x='Rating',
        palette='magma'
    )

    plt.xlabel('Rating')
    plt.ylabel('Number of Reviews')
    plt.title('Distribution of Restaurant Ratings')

    plt.show()

    print("\nRating distribution:")

    print(
        review['Rating']
        .value_counts(dropna=False)
        .sort_index()
    )


    # ========================================================
    # CHART 6 - TOP / BOTTOM AVERAGE RESTAURANT RATINGS
    # ========================================================

    restaurant_rating = (
        review
        .groupby('Restaurant')['Rating']
        .mean()
        .sort_values(ascending=False)
    )

    print(restaurant_rating.head(10))
    print(restaurant_rating.tail(10))


    top_10 = restaurant_rating.head(10).sort_values()
    bottom_10 = restaurant_rating.tail(10).sort_values()


    # Top 10

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=top_10.values,
        y=top_10.index
    )

    plt.xlabel('Average Rating')
    plt.ylabel('Restaurant')
    plt.title('Top 10 Restaurants by Average Rating')

    plt.xlim(0, 5)

    plt.show()


    # Bottom 10

    plt.figure(figsize=(10, 6))

    sns.barplot(
        x=bottom_10.values,
        y=bottom_10.index
    )

    plt.xlabel('Average Rating')
    plt.ylabel('Restaurant')
    plt.title('Bottom 10 Restaurants by Average Rating')

    plt.xlim(0, 5)

    plt.show()


    # Review count and rating summary

    review_count = review.groupby('Restaurant').size()

    rating_summary = pd.DataFrame({
        'Average_Rating': restaurant_rating,
        'Review_Count': review_count
    })

    print(
        rating_summary
        .sort_values(
            'Average_Rating',
            ascending=False
        )
        .head(10)
    )

    print(
        rating_summary
        .sort_values(
            'Average_Rating',
            ascending=True
        )
        .head(10)
    )


    # ========================================================
    # CHART 7 - RESTAURANT COST VS AVERAGE CUSTOMER RATING
    # ========================================================

    restaurant_rating = (
        review
        .groupby('Restaurant')['Rating']
        .mean()
        .reset_index()
    )

    rating_cost = restaurant_rating.merge(
        restaurant[['Name', 'Cost']],
        left_on='Restaurant',
        right_on='Name',
        how='inner'
    )


    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(
        rating_cost['Cost'],
        rating_cost['Rating'],
        s=60,
        c=rating_cost['Cost'],
        cmap='coolwarm',
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5
    )

    plt.colorbar(
        scatter,
        label='Restaurant Cost (INR)'
    )

    plt.xlabel('Restaurant Cost (INR)')
    plt.ylabel('Average Customer Rating')
    plt.title('Restaurant Cost vs Average Customer Rating')

    plt.show()


    # ========================================================
    # CHART 8 - NUMBER OF REVIEWS PER RESTAURANT
    # ========================================================

    review_count = review['Restaurant'].value_counts()

    print(review_count.head(10))


    plt.figure(figsize=(10, 6))

    sns.histplot(
        review_count,
        bins=20
    )

    plt.xlabel('Number of Reviews per Restaurant')
    plt.ylabel('Number of Restaurants')
    plt.title('Distribution of Reviews per Restaurant')

    plt.show()


    # ========================================================
    # CHART 9 - CUISINE COUNT VS AVERAGE RATING
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


    avg_rating_by_cuisine_count = (
        restaurant
        .groupby('Cuisine_Count')['Avg_Rating']
        .mean()
        .reset_index()
    )


    plt.figure(figsize=(10, 5))

    ax = sns.barplot(
        data=avg_rating_by_cuisine_count,
        x='Cuisine_Count',
        y='Avg_Rating',
        color='steelblue'
    )

    plt.title('Cuisine Count vs Average Rating')
    plt.xlabel('Number of Cuisines')
    plt.ylabel('Average Rating')
    plt.ylim(0, 5)

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt='%.2f'
        )

    plt.tight_layout()
    plt.show()


    # ========================================================
    # CHART 10 - AVERAGE COST BY NUMBER OF CUISINES
    # ========================================================

    avg_cost_by_cuisine_count = (
        restaurant
        .groupby('Cuisine_Count')['Cost']
        .mean()
        .reset_index()
    )


    plt.figure(figsize=(10, 5))

    ax = sns.barplot(
        data=avg_cost_by_cuisine_count,
        x='Cuisine_Count',
        y='Cost'
    )

    plt.title('Average Cost by Number of Cuisines')
    plt.xlabel('Number of Cuisines')
    plt.ylabel('Average Cost')

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt='₹%.0f'
        )

    plt.tight_layout()
    plt.show()


    # ========================================================
    # CHART 11 - MONTHLY REVIEW VOLUME OVER TIME
    # ========================================================

    monthly_reviews = (
        review
        .set_index('Time')
        .resample('ME')
        .size()
    )


    plt.figure(figsize=(12, 5))

    plt.plot(
        monthly_reviews.index,
        monthly_reviews.values,
        marker='o',
        markersize=5,
        linewidth=2
    )

    plt.xlabel('Months')
    plt.ylabel('Number of Reviews')
    plt.title('Monthly Review Volume')

    plt.grid(
        True,
        linestyle='--',
        alpha=0.5
    )

    plt.show()


    # ========================================================
    # CHART 12 - RATINGS OVER TIME
    # ========================================================

    plt.figure(figsize=(12, 5))

    review.set_index('Time')['Rating'].resample('ME').mean().plot()

    plt.xlabel('Time')
    plt.ylabel('Average Rating')
    plt.title('Average Restaurant Rating Over Time')

    plt.grid(
        True,
        linestyle='--',
        alpha=0.5
    )

    plt.show()


    # ========================================================
    # CHART 13 - POSITIVE VS NEGATIVE RATINGS
    # ========================================================

    rating_category = review.loc[
        review['Rating'].notna(),
        'Rating'
    ].apply(
        lambda x: 'Positive' if x >= 4 else 'Negative'
    )

    rating_counts = rating_category.value_counts()

    positive_count = rating_counts['Positive']
    negative_count = rating_counts['Negative']

    total_rated = positive_count + negative_count


    plt.figure(figsize=(8, 5))

    ax = sns.countplot(
        x=rating_category
    )

    ax.bar_label(
        ax.containers[0]
    )

    plt.xlabel('Rating Category')
    plt.ylabel('Number of Reviews')
    plt.title('Positive vs Negative Ratings')

    plt.show()


    positive_percentage = (
        positive_count / total_rated
    ) * 100

    negative_percentage = (
        negative_count / total_rated
    ) * 100


    print("Positive reviews:", positive_count)
    print("Negative reviews:", negative_count)
    print("Total rated reviews:", total_rated)

    print(
        f"Positive percentage: "
        f"{positive_percentage:.2f}%"
    )

    print(
        f"Negative percentage: "
        f"{negative_percentage:.2f}%"
    )


    # ========================================================
    # CHART 14 - CORRELATION HEATMAP
    # ========================================================

    review_agg = (
        review
        .groupby('Restaurant')
        .agg(
            Avg_Rating=('Rating', 'mean'),
            Avg_Pictures=('Pictures', 'mean')
        )
        .reset_index()
    )


    correlation_data = review_agg.merge(
        restaurant[['Name', 'Cost', 'Cuisine_Count']],
        left_on='Restaurant',
        right_on='Name',
        how='inner'
    )


    numeric_data = correlation_data[
        [
            'Cost',
            'Cuisine_Count',
            'Avg_Rating',
            'Avg_Pictures'
        ]
    ]


    correlation = numeric_data.corr()


    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidths=0.5
    )

    plt.title('Correlation Heatmap of Numeric Features')

    plt.show()


    # ========================================================
    # CHART 15 - PAIR PLOT
    # ========================================================

    pair_data = rating_cost.merge(
        restaurant[['Name', 'Cuisine_Count']],
        on='Name',
        how='left'
    )


    sns.pairplot(
        pair_data[
            [
                'Cost',
                'Cuisine_Count',
                'Rating'
            ]
        ],
        diag_kind='hist'
    )

    plt.suptitle(
        "Pair Plot — Restaurant-Level Features",
        y=1.01,
        fontsize=13
    )

    plt.show()


    # ========================================================
    # RETURN UPDATED DATA
    # ========================================================

    return restaurant, review