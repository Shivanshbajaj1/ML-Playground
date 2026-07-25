import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)


def show_classification_results(y_test, predictions):

    st.subheader("📊 Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )


    st.subheader("🔥 Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)