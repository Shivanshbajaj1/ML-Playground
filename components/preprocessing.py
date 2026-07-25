import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """
    Handle target column selection, preprocessing, and train-test split.
    This function contains the exact code from the original app.py for preprocessing.

    Args:
        df: pandas.DataFrame (the loaded dataset)

    Returns:
        tuple: (X_train, X_test, y_train, y_test, target, problem_type, X_columns)
               or (None, None, None, None, None, None, None) if no target selected
    """
    st.subheader("Target Column")
    target = st.selectbox("Select the target column", df.columns)
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    numeric_columns = X.select_dtypes(include=["number"]).columns
    categorical_columns = X.select_dtypes(include=["object"]).columns

    X[numeric_columns] = X[numeric_columns].fillna(
        X[numeric_columns].mean()
    )

    X[categorical_columns] = X[categorical_columns].fillna("Unknown")

    st.success(f"Selected Target: {target}")

    st.subheader("Features")
    feature_df = pd.DataFrame({"Feature": X.columns})
    st.dataframe(feature_df, use_container_width=True)

    st.subheader("Problem Type")
    if y.dtype == "object" or y.nunique() <= 10:
        problem_type = "Classification"
    else:
        problem_type = "Regression"

    st.success(problem_type)

    st.subheader("Categorical Columns")
    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()

    if categorical_columns:
        st.write(categorical_columns)
        X = pd.get_dummies(X, columns=categorical_columns)
        st.success("Categorical columns encoded successfully!")
    else:
        st.success("No categorical columns found.")

    st.subheader("Processed Features")
    st.dataframe(X.head(), use_container_width=True)

    st.subheader("Train-Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Samples", X_train.shape[0])
    with col2:
        st.metric("Testing Samples", X_test.shape[0])

    # Return the processed data and metadata
    return X_train, X_test, y_train, y_test, target, problem_type, list(X.columns)