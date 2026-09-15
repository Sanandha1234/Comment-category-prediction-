import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.sparse import hstack, csr_matrix


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Comment Category Predictor",
    page_icon="💬",
    layout="wide",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .section-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e5e7eb;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">💬 Comment Category Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "An NLP + machine learning application for classifying online comments."
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# PROJECT OVERVIEW
# =========================================================

with st.expander("ℹ️ About this project", expanded=False):

    st.write(
        """
        This application predicts the category of an online comment using
        a trained LightGBM classification model.

        The model combines:

        • Word-level TF-IDF features  
        • Character-level TF-IDF features  
        • Comment statistics  
        • Voting information  
        • Emoticon indicators  
        • Other engineered numerical features
        """
    )

    st.info(
        "The deployed model is LightGBM, selected after comparing "
        "LinearSVC, Logistic Regression, and LightGBM."
    )


# =========================================================
# LOAD ARTIFACTS
# =========================================================

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "lgbm_full_model.pkl"
WORD_PATH = ARTIFACT_DIR / "vectorizer_word.pkl"
CHAR_PATH = ARTIFACT_DIR / "vectorizer_char.pkl"
SCALER_PATH = ARTIFACT_DIR / "scaler (1).pkl"


@st.cache_resource
def load_artifacts():

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(WORD_PATH, "rb") as f:
        word_vectorizer = pickle.load(f)

    with open(CHAR_PATH, "rb") as f:
        char_vectorizer = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, word_vectorizer, char_vectorizer, scaler


model, word_vectorizer, char_vectorizer, scaler = load_artifacts()


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def make_numeric_features(
    comment,
    upvote,
    downvote,
    if_1,
    if_2,
    emoticon_1,
    emoticon_2,
    emoticon_3,
    disability,
):

    comment = str(comment)

    comment_length = len(comment)
    word_count = len(comment.split())
    exclamation_count = comment.count("!")
    question_count = comment.count("?")

    uppercase_count = sum(
        1 for char in comment if char.isupper()
    )

    capital_ratio = uppercase_count / (comment_length + 1)

    total_emoticons = (
        emoticon_1
        + emoticon_2
        + emoticon_3
    )

    vote_diff = upvote - downvote
    total_votes = upvote + downvote

    values = [[
        upvote,
        downvote,
        if_1,
        if_2,
        emoticon_1,
        emoticon_2,
        emoticon_3,
        disability,
        comment_length,
        word_count,
        exclamation_count,
        question_count,
        capital_ratio,
        total_emoticons,
        vote_diff,
        total_votes,
    ]]

    return np.asarray(values, dtype=float)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_comment(
    model,
    word_vectorizer,
    char_vectorizer,
    scaler,
    comment,
    upvote,
    downvote,
    if_1,
    if_2,
    emoticon_1,
    emoticon_2,
    emoticon_3,
    disability,
):

    comment_text = [str(comment)]

    # Word TF-IDF
    word_features = word_vectorizer.transform(
        comment_text
    )

    # Character TF-IDF
    char_features = char_vectorizer.transform(
        comment_text
    )

    # Numerical features
    numeric_features = make_numeric_features(
        comment,
        upvote,
        downvote,
        if_1,
        if_2,
        emoticon_1,
        emoticon_2,
        emoticon_3,
        disability,
    )

    numeric_scaled = scaler.transform(
        numeric_features
    )

    numeric_sparse = csr_matrix(
        numeric_scaled
    )

    # Combine all features
    all_features = hstack(
        [
            word_features,
            char_features,
            numeric_sparse
        ],
        format="csr",
    )

    prediction = model.predict(
        all_features
    )[0]

    probabilities = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            all_features
        )[0]

    return prediction, probabilities


# =========================================================
# DEMO EXAMPLES
# =========================================================

EXAMPLES = {

    "Choose a demo example": {
        "comment": "",
        "upvote": 0,
        "downvote": 0,
        "if_1": 0,
        "if_2": 0,
        "emoticon_1": 0,
        "emoticon_2": 0,
        "emoticon_3": 0,
        "disability": 0,
    },

    "Friendly comment 😊": {
        "comment": "Thank you so much 😊❤️ You made my day! 👍",
        "upvote": 12,
        "downvote": 1,
        "if_1": 0,
        "if_2": 0,
        "emoticon_1": 1,
        "emoticon_2": 1,
        "emoticon_3": 1,
        "disability": 0,
    },

    "Critical comment": {
        "comment": "I completely disagree with this opinion.",
        "upvote": 3,
        "downvote": 6,
        "if_1": 0,
        "if_2": 0,
        "emoticon_1": 0,
        "emoticon_2": 0,
        "emoticon_3": 0,
        "disability": 0,
    },

    "Insulting comment": {
        "comment": "You are so stupid and useless!",
        "upvote": 5,
        "downvote": 2,
        "if_1": 0,
        "if_2": 0,
        "emoticon_1": 0,
        "emoticon_2": 0,
        "emoticon_3": 0,
        "disability": 0,
    },

    "Neutral comment": {
        "comment": "The meeting starts at 3 PM.",
        "upvote": 0,
        "downvote": 0,
        "if_1": 0,
        "if_2": 0,
        "emoticon_1": 0,
        "emoticon_2": 0,
        "emoticon_3": 0,
        "disability": 0,
    },

    "Emoji-heavy comment": {
        "comment": "Amazing work!!! 😍🔥👏❤️",
        "upvote": 25,
        "downvote": 1,
        "if_1": 0,
        "if_2": 0,
        "emoticon_1": 2,
        "emoticon_2": 2,
        "emoticon_3": 1,
        "disability": 0,
    },
}


# =========================================================
# SESSION STATE
# =========================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "example_values" not in st.session_state:
    st.session_state.example_values = (
        EXAMPLES["Choose a demo example"].copy()
    )


def load_example():

    selected = st.session_state.example_selector

    st.session_state.example_values = (
        EXAMPLES[selected].copy()
    )


# =========================================================
# DEMO SELECTOR
# =========================================================

st.subheader("🧪 Try a demo comment")

st.selectbox(
    "Choose an example",
    list(EXAMPLES.keys()),
    key="example_selector",
    on_change=load_example,
)

values = st.session_state.example_values


# =========================================================
# INPUT FORM
# =========================================================

with st.form("prediction_form"):

    st.subheader("📝 Comment")

    comment = st.text_area(
        "Comment text",
        value=values["comment"],
        placeholder="Type or paste a comment here…",
        height=140,
    )

    st.subheader("📋 Metadata")

    st.caption(
        "Enter the metadata associated with the comment."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        upvote = st.number_input(
            "👍 Upvotes",
            min_value=0,
            step=1,
            value=int(values["upvote"]),
        )

        if_1 = st.number_input(
            "if_1",
            min_value=0,
            step=1,
            value=int(values["if_1"]),
        )

        emoticon_1 = st.number_input(
            "😊 Emoticon 1",
            min_value=0,
            step=1,
            value=int(values["emoticon_1"]),
        )

    with col2:

        downvote = st.number_input(
            "👎 Downvotes",
            min_value=0,
            step=1,
            value=int(values["downvote"]),
        )

        if_2 = st.number_input(
            "if_2",
            min_value=0,
            step=1,
            value=int(values["if_2"]),
        )

        emoticon_2 = st.number_input(
            "❤️ Emoticon 2",
            min_value=0,
            step=1,
            value=int(values["emoticon_2"]),
        )

    with col3:

        disability = st.number_input(
            "Disability flag",
            min_value=0,
            step=1,
            value=int(values["disability"]),
        )

        emoticon_3 = st.number_input(
            "🔥 Emoticon 3",
            min_value=0,
            step=1,
            value=int(values["emoticon_3"]),
        )

    submitted = st.form_submit_button(
        "🔍 Predict Category",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# RUN PREDICTION
# =========================================================

if submitted:

    if not comment.strip():

        st.warning(
            "Please enter a comment before predicting."
        )

    else:

        try:

            with st.spinner(
                "Analyzing comment..."
            ):

                prediction, probabilities = predict_comment(
                    model,
                    word_vectorizer,
                    char_vectorizer,
                    scaler,
                    comment,
                    int(upvote),
                    int(downvote),
                    int(if_1),
                    int(if_2),
                    int(emoticon_1),
                    int(emoticon_2),
                    int(emoticon_3),
                    int(disability),
                )

            st.session_state.last_prediction = prediction
            st.session_state.last_scores = probabilities
            st.session_state.last_comment = comment

            st.session_state.prediction_history.append(
                {
                    "Comment": comment,
                    "Predicted category": str(prediction),
                }
            )

            st.success(
                "Prediction completed successfully!"
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# =========================================================
# PREDICTION RESULT
# =========================================================

if "last_prediction" in st.session_state:

    st.divider()

    st.subheader("🎯 Prediction Result")

    prediction = st.session_state.last_prediction
    probabilities = st.session_state.last_scores

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Predicted Category",
            str(prediction),
        )

    with result_col2:

        if probabilities is not None:

            confidence = float(
                np.max(probabilities)
            )

            st.metric(
                "Model Confidence",
                f"{confidence:.1%}",
            )

    st.caption(
        "The predicted category is the model's numeric label."
    )


# =========================================================
# MODEL PROBABILITIES
# =========================================================

if "last_scores" in st.session_state:

    probabilities = st.session_state.last_scores

    if probabilities is not None:

        st.subheader("📊 Model Probability Distribution")

        score_df = pd.DataFrame(
            {
                "Category": [
                    str(label)
                    for label in model.classes_
                ],
                "Probability": probabilities,
            }
        )

        score_df = score_df.sort_values(
            "Probability",
            ascending=False,
        )

        score_display = score_df.copy()

        score_display["Probability"] = (
            score_display["Probability"]
            .map(lambda x: f"{x:.2%}")
        )

        st.dataframe(
            score_display,
            use_container_width=True,
            hide_index=True,
        )

        st.bar_chart(
            score_df.set_index("Category"),
            y="Probability",
        )


# =========================================================
# COMMENT ANALYTICS
# =========================================================

if "last_prediction" in st.session_state:

    st.divider()

    st.subheader("🔎 Comment Analytics")

    current_comment = st.session_state.last_comment

    comment_length = len(current_comment)
    word_count = len(current_comment.split())
    exclamation_count = current_comment.count("!")
    question_count = current_comment.count("?")

    uppercase_count = sum(
        1
        for char in current_comment
        if char.isupper()
    )

    capital_ratio = (
        uppercase_count /
        (comment_length + 1)
    )

    total_emoticons = (
        int(emoticon_1)
        + int(emoticon_2)
        + int(emoticon_3)
    )

    vote_diff = (
        int(upvote)
        - int(downvote)
    )

    total_votes = (
        int(upvote)
        + int(downvote)
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Characters",
            comment_length,
        )

    with c2:
        st.metric(
            "Words",
            word_count,
        )

    with c3:
        st.metric(
            "Total Votes",
            total_votes,
        )

    with c4:
        st.metric(
            "Emoticons",
            total_emoticons,
        )

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        st.metric(
            "👍 Upvotes",
            int(upvote),
        )

    with c6:
        st.metric(
            "👎 Downvotes",
            int(downvote),
        )

    with c7:
        st.metric(
            "❗ Exclamations",
            exclamation_count,
        )

    with c8:
        st.metric(
            "❓ Questions",
            question_count,
        )

    st.write(
        f"**Capitalization ratio:** "
        f"{capital_ratio:.2%}"
    )

    st.write(
        f"**Vote difference:** {vote_diff}"
    )


# =========================================================
# SESSION ANALYTICS
# =========================================================

st.divider()

st.subheader("📈 Session Analytics")

history = st.session_state.prediction_history

if history:

    history_df = pd.DataFrame(history)

    category_counts = (
        history_df[
            "Predicted category"
        ]
        .value_counts()
        .sort_index()
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Predictions",
            len(history_df),
        )

    with col2:

        st.metric(
            "Categories Seen",
            history_df[
                "Predicted category"
            ].nunique(),
        )

    with col3:

        most_common = (
            category_counts.idxmax()
        )

        st.metric(
            "Most Predicted",
            most_common,
        )

    st.write(
        "Predicted category distribution"
    )

    st.bar_chart(
        category_counts
    )

    with st.expander(
        "📜 View prediction history"
    ):

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

    csv_data = (
        history_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        "⬇️ Download Prediction History",
        data=csv_data,
        file_name="comment_prediction_history.csv",
        mime="text/csv",
    )

    if st.button(
        "🗑️ Clear Session History"
    ):

        st.session_state.prediction_history = []

        st.rerun()

else:

    st.info(
        "Make a prediction to generate session analytics."
    )


# =========================================================
# MODEL COMPARISON
# =========================================================

st.divider()

st.subheader("📈 Model Comparison")

st.write(
    """
    Three classification approaches were evaluated during model
    development. The models were compared using weighted F1 score.
    LightGBM achieved the strongest validation performance and was
    selected as the final deployed model.
    """
)

comparison_image = (
    Path(__file__).parent
    / "assets"
    / "model_comparison.png"
)

if comparison_image.exists():

    st.image(
        str(comparison_image),
        caption="Weighted F1 Score Comparison",
        use_container_width=True,
    )

else:

    st.warning(
        "model_comparison.png was not found inside the assets folder."
    )


with st.expander(
    "🔬 Why was LightGBM selected?"
):

    st.markdown(
        """
        **LinearSVC**

        A strong baseline for high-dimensional sparse TF-IDF features.

        **Logistic Regression**

        Provides a probabilistic linear classification approach and
        performed competitively on the text representation.

        **LightGBM**

        Captured non-linear relationships between the text-derived
        representation and engineered numerical features.

        Since LightGBM achieved the highest weighted F1 score in the
        model comparison, it was selected for deployment.
        """
    )


# =========================================================
# MODEL PIPELINE
# =========================================================

st.divider()

st.subheader("🧠 How the Model Works")

pipeline_col1, pipeline_col2, pipeline_col3 = st.columns(3)

with pipeline_col1:

    st.markdown(
        """
        ### 1️⃣ Text Features

        **Word TF-IDF**

        Captures important words and word combinations.

        **Character TF-IDF**

        Captures character-level patterns and variations.
        """
    )

with pipeline_col2:

    st.markdown(
        """
        ### 2️⃣ Engineered Features

        • Comment length  
        • Word count  
        • Upvotes / downvotes  
        • Vote difference  
        • Total votes  
        • Punctuation counts  
        • Capitalization ratio  
        • Emoticon features
        """
    )

with pipeline_col3:

    st.markdown(
        """
        ### 3️⃣ Final Prediction

        Word TF-IDF + Character TF-IDF + numerical features

        ↓

        **LightGBM**

        ↓

        **Predicted Category**
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Comment Category Prediction • NLP • TF-IDF • LightGBM • Streamlit"
)