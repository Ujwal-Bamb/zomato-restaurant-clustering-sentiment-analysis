# ============================================================
# ZOMATO SENTIMENT ANALYSIS
# STREAMLIT APPLICATION
# ============================================================

import streamlit as st
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Zomato Sentiment Analysis",
    page_icon="🍽️",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🍽️ Zomato Restaurant Sentiment Analysis")

st.write(
    "Enter a restaurant review below and the trained "
    "Logistic Regression model will predict the sentiment."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_package = joblib.load(
        "models/zomato_sentiment_model.joblib"
    )

    model = model_package["model"]
    tfidf = model_package["tfidf"]

    return model, tfidf


model, tfidf = load_model()


# ============================================================
# REVIEW INPUT
# ============================================================

review_text = st.text_area(
    "Enter your restaurant review:",
    placeholder="Example: The food was excellent and the service was amazing.",
    height=150
)


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "Predict Sentiment",
    type="primary"
):

    if review_text.strip() == "":

        st.warning(
            "Please enter a review before making a prediction."
        )

    else:

        # ----------------------------------------------------
        # Convert review into TF-IDF
        # ----------------------------------------------------

        review_vector = tfidf.transform(
            [review_text]
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            review_vector
        )[0]

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        st.subheader("Prediction")

        if prediction == "Positive":

            st.success(
                "😊 Positive Sentiment"
            )

        else:

            st.error(
                "😞 Negative Sentiment"
            )