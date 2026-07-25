import streamlit as st
import pandas as pd

def load_and_display_data(uploaded_file):
    """
    Load and display basic information about the uploaded CSV file.
    This function contains the exact code from the original app.py for data loading and display.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        pandas.DataFrame: The loaded dataframe or None if no file uploaded
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")
        st.dataframe(df, use_container_width=True)

        rows, columns = df.shape

        st.subheader("Dataset Shape")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", rows)
        with col2:
            st.metric("Columns", columns)

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("Missing Values")
        st.dataframe(
            df.isnull().sum().to_frame("Missing Values"), use_container_width=True
        )

        st.subheader("Column Information")
        column_info = pd.DataFrame(
            {"Column": df.columns, "Data Type": df.dtypes.astype(str)}
        )
        st.dataframe(column_info, use_container_width=True)

        st.subheader("Duplicate Rows")
        duplicates = df.duplicated().sum()
        st.write(f"Total Duplicate Rows: {duplicates}")

        if duplicates > 0:
            st.dataframe(df[df.duplicated()], use_container_width=True)
        else:
            st.success("No duplicate rows found!")

        st.subheader("Memory Usage")
        memory = df.memory_usage(deep=True).sum() / 1024
        st.write(f"{memory:.2f} KB")

        return df
    return None