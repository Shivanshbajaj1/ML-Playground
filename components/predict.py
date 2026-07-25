import streamlit as st
import pandas as pd


def predict_new_data(model, X_columns):

    st.subheader("🔮 Predict New Data")

    user_input = {}

    for column in X_columns:
        user_input[column] = st.number_input(
            column,
            value=0.0
        )

    if st.button("Predict"):

        input_df = pd.DataFrame([user_input])

        prediction = model.predict(input_df)

        st.success(f"Prediction: {prediction[0]}")