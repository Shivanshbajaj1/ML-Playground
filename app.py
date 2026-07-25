import streamlit as st
from components.data_loader import load_and_display_data
from components.preprocessing import preprocess_data
from components.modeling import train_selected_model, compare_all_models
from components.evaluation import show_classification_results
from components.predict import predict_new_data

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="ML Playground",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ML Playground")
st.caption(
    "Train • Compare • Evaluate • Download Machine Learning Models"
)
with st.sidebar:
    st.title("🤖 ML Playground")

    st.markdown("---")

    st.write("### Features")

    st.success("✅ Upload CSV")
    st.success("✅ Automatic Preprocessing")
    st.success("✅ Multiple ML Models")
    st.success("✅ Cross Validation")
    st.success("✅ Feature Importance")
    st.success("✅ Model Download")
    st.success("✅ Prediction Export")

    st.markdown("---")

    st.write("### About")

    st.info(
        """
        ML Playground is an interactive machine learning web app built with
        Streamlit and Scikit-learn.

        Developed by Shivansh Bajaj.
        """
    )
st.write("Upload a dataset and train Machine Learning models.")

# ----------------------------
# Upload Dataset
# ----------------------------
uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

df = load_and_display_data(uploaded_file)

# ----------------------------
# Main App
# ----------------------------
if df is not None:

    result = preprocess_data(df)

    if result[0] is not None:

        (
            X_train,
            X_test,
            y_train,
            y_test,
            target,
            problem_type,
            X_columns,
        ) = result

        # Create Tabs
        tab1, tab2, tab3 = st.tabs(
            ["📂 Data", "🤖 Training", "📊 Results"]
        )

        # ==================================================
        # TAB 1 : DATA
        # ==================================================
        with tab1:

            st.subheader("Dataset Preview")

            st.dataframe(
                df.head(),
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Rows", df.shape[0])

            with col2:
                st.metric("Columns", df.shape[1])

            st.write("### Selected Target")
            st.info(target)

            st.write("### Problem Type")
            st.success(problem_type)

        # ==================================================
        # TAB 2 : TRAINING
        # ==================================================
        with tab2:

            st.subheader("Choose Machine Learning Model")

            if problem_type == "Classification":

                model_name = st.selectbox(
                    "Model",
                    [
                        "Logistic Regression",
                        "Decision Tree",
                        "Random Forest",
                        "K-Nearest Neighbors",
                    ],
                )

            else:

                model_name = st.selectbox(
                    "Model",
                    [
                        "Linear Regression",
                        "Decision Tree",
                        "Random Forest",
                        "K-Nearest Neighbors",
                    ],
                )

            col1, col2 = st.columns(2)

            with col1:
                train_btn = st.button(
                    "🚀 Train Selected Model"
                )

            with col2:
                compare_btn = st.button(
                    "📊 Compare All Models"
                )

        # ==================================================
        # TAB 3 : RESULTS
        # ==================================================
        with tab3:

            if train_btn:

                with st.spinner("Training model..."):

                    model, predictions = train_selected_model(
                        problem_type,
                        model_name,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        X_columns,
                    )

                predict_new_data(
                                model,
                                X_columns
                                )
                st.success("Model trained successfully! 🎉")

                if problem_type == "Classification":
                    show_classification_results(
                        y_test,
                        predictions
                    )

            if compare_btn:

                with st.spinner("Comparing models..."):

                    compare_all_models(
                        problem_type,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                    )

else:

    st.info("Please upload a CSV file to begin.")