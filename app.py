import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st

from src.config import (
    RESULTS_DIR,
    DEVICE,
)

from src.inference import (
    load_all_models,
    predict_image,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Indian Bird AI",
    page_icon=None,
    layout="wide",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.7rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 2rem;
    }

    .prediction-card {
        padding: 1.4rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        margin-top: 1rem;
    }

    .species-name {
        font-size: 2rem;
        font-weight: 700;
    }

    .confidence {
        font-size: 1.25rem;
        margin-top: 0.4rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# CACHE ML MODELS
# =========================================================

@st.cache_resource(
    show_spinner="Loading four deep-learning models..."
)
def get_models():
    """
    Load all four trained models once.

    Streamlit reruns app.py whenever widgets change,
    therefore models must be cached instead of loaded
    repeatedly.
    """

    return load_all_models()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def prettify_species(name):
    """
    Converts:
        Indian-Peacock

    into:
        Indian Peacock
    """

    return name.replace("-", " ")


def confidence_dataframe(predictions):
    """
    Convert top predictions into a table.
    """

    return pd.DataFrame(
        [
            {
                "Species": prettify_species(
                    item["class"]
                ),
                "Confidence": (
                    item["confidence"] * 100
                ),
            }
            for item in predictions
        ]
    )


def uploaded_file_hash(uploaded_file):
    """
    Detect when the user uploads a different image.
    """

    return hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "Indian Bird AI"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Predict",
        "Model Performance",
        "Confusion Matrices",
        "Methodology",
    ],
)

st.sidebar.divider()

st.sidebar.write(
    f"Inference device: `{DEVICE}`"
)

st.sidebar.caption(
    "25 Indian bird species"
)


# =========================================================
# PREDICTION PAGE
# =========================================================

if page == "Predict":

    st.markdown(
        '<div class="main-title">'
        'Indian Bird Species Classifier'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Multi-model deep learning classification using '
        'ResNet50, EfficientNetV2-B0, MobileNetV2, '
        'ViT-B/32 and ensemble learning.'
        '</div>',
        unsafe_allow_html=True,
    )

    left_column, right_column = st.columns(
        [1, 1.2]
    )

    # -----------------------------------------------------
    # IMAGE UPLOAD
    # -----------------------------------------------------

    with left_column:

        st.subheader(
            "Upload Bird Image"
        )

        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG or PNG image",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
        )

        if uploaded_file is not None:

            image_bytes = (
                uploaded_file.getvalue()
            )

            image = Image.open(
                BytesIO(image_bytes)
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded image",
                width="stretch",
            )

            current_hash = (
                uploaded_file_hash(
                    uploaded_file
                )
            )

            # If a new image was uploaded,
            # discard prediction from previous image.

            if (
                st.session_state.get(
                    "image_hash"
                )
                != current_hash
            ):

                st.session_state[
                    "image_hash"
                ] = current_hash

                st.session_state.pop(
                    "prediction_results",
                    None,
                )

            classify_button = (
                st.button(
                    "Classify Image",
                    type="primary",
                    width="stretch",
                )
            )

            if classify_button:

                models = get_models()

                with st.spinner(
                    "Running all four models and ensembles..."
                ):

                    results = predict_image(
                        image,
                        models,
                    )

                st.session_state[
                    "prediction_results"
                ] = results

    # -----------------------------------------------------
    # PRIMARY RESULT
    # -----------------------------------------------------

    with right_column:

        st.subheader(
            "Prediction"
        )

        results = st.session_state.get(
            "prediction_results"
        )

        if results is None:

            st.info(
                "Upload an image and select "
                "'Classify Image' to generate a prediction."
            )

        else:

            geometric_results = results[
                "Geometric Mean Ensemble"
            ]

            best_prediction = (
                geometric_results[0]
            )

            species = prettify_species(
                best_prediction["class"]
            )

            confidence = (
                best_prediction[
                    "confidence"
                ]
                * 100
            )

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div>
                        Geometric Mean Ensemble
                    </div>

                    <div class="species-name">
                        {species}
                    </div>

                    <div class="confidence">
                        Confidence: {confidence:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader(
                "Top-3 Predictions"
            )

            top3_dataframe = (
                confidence_dataframe(
                    geometric_results
                )
            )

            st.dataframe(
                top3_dataframe,
                hide_index=True,
                width="stretch",
                column_config={
                    "Confidence":
                        st.column_config.ProgressColumn(
                            "Confidence (%)",
                            min_value=0,
                            max_value=100,
                            format="%.2f%%",
                        )
                },
            )

    # -----------------------------------------------------
    # MODEL-BY-MODEL RESULTS
    # -----------------------------------------------------

    results = st.session_state.get(
        "prediction_results"
    )

    if results is not None:

        st.divider()

        st.header(
            "Model Comparison"
        )

        rows = []

        for model_name, predictions in (
            results.items()
        ):

            best = predictions[0]

            rows.append(
                {
                    "Model": model_name,
                    "Prediction":
                        prettify_species(
                            best["class"]
                        ),
                    "Confidence (%)":
                        best["confidence"]
                        * 100,
                }
            )

        comparison = pd.DataFrame(
            rows
        )

        st.dataframe(
            comparison,
            hide_index=True,
            width="stretch",
            column_config={
                "Confidence (%)":
                    st.column_config.ProgressColumn(
                        "Confidence (%)",
                        min_value=0,
                        max_value=100,
                        format="%.2f%%",
                    )
            },
        )

        st.subheader(
            "Top-3 by Model"
        )

        model_choice = st.selectbox(
            "Select model",
            list(
                results.keys()
            ),
        )

        selected_predictions = (
            confidence_dataframe(
                results[model_choice]
            )
        )

        st.dataframe(
            selected_predictions,
            hide_index=True,
            width="stretch",
        )


# =========================================================
# PERFORMANCE PAGE
# =========================================================

elif page == "Model Performance":

    st.title(
        "Model Performance"
    )

    metrics_path = (
        RESULTS_DIR
        / "metrics.csv"
    )

    if not metrics_path.exists():

        st.error(
            "results/metrics.csv was not found."
        )

    else:

        metrics = pd.read_csv(
            metrics_path
        )

        metric_columns = [
            "Accuracy",
            "Top-3 Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC",
        ]

        # Convert decimal metrics to percentages.
        display_metrics = (
            metrics.copy()
        )

        for column in metric_columns:

            display_metrics[column] = (
                display_metrics[column]
                * 100
            )

        best_row = (
            metrics.loc[
                metrics["Accuracy"].idxmax()
            ]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Best Model",
            best_row["Model"],
        )

        col2.metric(
            "Best Accuracy",
            f"{best_row['Accuracy'] * 100:.2f}%",
        )

        col3.metric(
            "Top-3 Accuracy",
            f"{best_row['Top-3 Accuracy'] * 100:.2f}%",
        )

        st.subheader(
            "Complete Results"
        )

        formatted = (
            display_metrics.copy()
        )

        for column in metric_columns:

            formatted[column] = (
                formatted[column]
                .map(
                    lambda value:
                    f"{value:.2f}%"
                )
            )

        st.dataframe(
            formatted,
            hide_index=True,
            width="stretch",
        )

        st.subheader(
            "Accuracy Comparison"
        )

        accuracy_chart = (
            display_metrics[
                [
                    "Model",
                    "Accuracy",
                ]
            ]
            .set_index("Model")
        )

        st.bar_chart(
            accuracy_chart
        )

        st.subheader(
            "F1 Score Comparison"
        )

        f1_chart = (
            display_metrics[
                [
                    "Model",
                    "F1 Score",
                ]
            ]
            .set_index("Model")
        )

        st.bar_chart(
            f1_chart
        )


# =========================================================
# CONFUSION MATRIX PAGE
# =========================================================

elif page == "Confusion Matrices":

    st.title(
        "Confusion Matrices"
    )

    matrix_directory = (
        RESULTS_DIR
        / "confusion_matrices"
    )

    matrix_files = {
        "ResNet50":
            "resnet50_confusion_matrix.png",

        "EfficientNetV2-B0":
            "efficientnet_v2_b0_confusion_matrix.png",

        "MobileNetV2":
            "mobilenet_v2_confusion_matrix.png",

        "ViT-B/32":
            "vit_b32_confusion_matrix.png",

        "Linear Mean Ensemble":
            "linear_mean_ensemble_confusion_matrix.png",

        "Geometric Mean Ensemble":
            "geometric_mean_ensemble_confusion_matrix.png",
    }

    selected_model = st.selectbox(
        "Select model",
        list(
            matrix_files.keys()
        ),
    )

    matrix_path = (
        matrix_directory
        / matrix_files[selected_model]
    )

    if matrix_path.exists():

        st.image(
            str(matrix_path),
            caption=(
                f"{selected_model} "
                "Confusion Matrix"
            ),
            width="stretch",
        )

    else:

        st.warning(
            f"Confusion matrix not found: "
            f"{matrix_path.name}"
        )


# =========================================================
# METHODOLOGY PAGE
# =========================================================

elif page == "Methodology":

    st.title(
        "Methodology"
    )

    st.header(
        "Dataset"
    )

    st.write(
        """
        The project classifies 25 Indian bird species
        from a dataset containing 37,500 images.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Training Images",
        "24,000",
    )

    col2.metric(
        "Validation Images",
        "7,500",
    )

    col3.metric(
        "Test Images",
        "6,000",
    )

    st.header(
        "Transfer Learning"
    )

    st.write(
        """
        Four pretrained image-classification architectures
        are adapted to the 25-class classification problem.
        Their pretrained feature extractors are used with
        newly trained classification heads.
        """
    )

    st.write(
        """
        - ResNet50
        - EfficientNetV2-B0
        - MobileNetV2
        - Vision Transformer ViT-B/32
        """
    )

    st.header(
        "Ensemble Learning"
    )

    st.subheader(
        "Linear Mean"
    )

    st.latex(
        r"""
        p_{\mathrm{linear}}
        =
        \frac{
            p_1 + p_2 + p_3 + p_4
        }{4}
        """
    )

    st.subheader(
        "Geometric Mean"
    )

    st.latex(
        r"""
        p_{\mathrm{geometric}}
        =
        \left(
            p_1 p_2 p_3 p_4
        \right)^{1/4}
        """
    )

    st.header(
        "Evaluation Metrics"
    )

    st.write(
        """
        Models are evaluated using Accuracy,
        Top-3 Accuracy, Precision, Recall,
        F1 Score and Matthews Correlation Coefficient.
        """
    )