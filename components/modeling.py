import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score


def train_selected_model(problem_type, model_name, X_train, y_train, X_test, y_test, X_columns):
    """Train the selected model and display results."""

    # 1. Instantiate model
    if problem_type == "Classification":
        if model_name == "Logistic Regression":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000))
            ])
        elif model_name == "K-Nearest Neighbors":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier())
            ])
        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        else:
            model = LogisticRegression(max_iter=1000)
    else:  # Regression
        if model_name == "Linear Regression":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression())
            ])
        elif model_name == "K-Nearest Neighbors":
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", KNeighborsRegressor())
            ])
        elif model_name == "Decision Tree":
            model = DecisionTreeRegressor(random_state=42)
        elif model_name == "Random Forest":
            model = RandomForestRegressor(random_state=42)
        else:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression())
            ])

    # 2. Cross Validation & Training
    model.fit(X_train, y_train)

    if problem_type == "Classification":
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy"
        )
    else:
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="r2"
        )

    st.subheader("📈 Cross Validation")

    metric_label = "Accuracy" if problem_type == "Classification" else "R² Score"
    avg_score_fmt = f"{cv_scores.mean():.2%}" if problem_type == "Classification" else f"{cv_scores.mean():.4f}"

    st.write(f"Average CV {metric_label}: **{avg_score_fmt}**")

    cv_df = pd.DataFrame({
        "Fold": range(1, len(cv_scores) + 1),
        metric_label: cv_scores
    })
    st.dataframe(cv_df, use_container_width=True)

    predictions = model.predict(X_test)

    # 3. Save Model
    model_path = "models/trained_model.pkl"
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)

    # 4. Display Test Metrics
    col1, col2 = st.columns(2)
    with col1:
        if problem_type == "Classification":
            accuracy = accuracy_score(y_test, predictions)
            st.metric("Test Accuracy", f"{accuracy:.2%}")
        else:
            score = r2_score(y_test, predictions)
            st.metric("Test R² Score", f"{score:.4f}")

    with col2:
        st.metric("Selected Model", model_name)

    # 5. Predictions Preview & Download
    prediction_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })
    st.subheader("Predictions Preview")
    st.dataframe(prediction_df.head(10), use_container_width=True)
    
    csv = prediction_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Predictions CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )
    st.success("Model trained and saved successfully!")

    # 6. Model Download Button
    if os.path.exists(model_path):
        with open(model_path, "rb") as file:
            st.download_button(
                label="💾 Download Trained Model",
                data=file,
                file_name="trained_model.pkl",
                mime="application/octet-stream"
            )
    else:
        st.warning("Model file was not created.")

    # 7. Feature Importance (if applicable)
    final_model = model.steps[-1][1] if isinstance(model, Pipeline) else model

    if hasattr(final_model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": X_columns,
            "Importance": final_model.feature_importances_,
        }).sort_values(by="Importance", ascending=False)

        st.subheader("Feature Importance")
        st.dataframe(importance_df, use_container_width=True)

        fig = px.bar(
            importance_df.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Important Features",
        )

        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    return model, predictions


def compare_all_models(problem_type, X_train, y_train, X_test, y_test):
    """Compare all models and display results."""
    if problem_type == "Classification":
        models = {
            "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
        }

        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            results.append({
                "Model": name,
                "Accuracy": round(accuracy * 100, 2)
            })

        results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
        results_df.insert(0, "Rank", range(1, len(results_df) + 1))

        st.subheader("🏆 Model Leaderboard")
        st.dataframe(results_df, use_container_width=True)

        best_model = results_df.iloc[0]
        st.success(f"Best Model: {best_model['Model']} ({best_model['Accuracy']}%)")

        fig = px.bar(
            results_df,
            x="Model",
            y="Accuracy",
            color="Model",
            text="Accuracy",
            title="Model Accuracy Comparison",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    else:
        models = {
            "Linear Regression": Pipeline([("scaler", StandardScaler()), ("reg", LinearRegression())]),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "KNN": Pipeline([("scaler", StandardScaler()), ("reg", KNeighborsRegressor())]),
        }

        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = r2_score(y_test, predictions)
            results.append({
                "Model": name,
                "R² Score": round(score, 4)
            })

        results_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False)
        results_df.insert(0, "Rank", range(1, len(results_df) + 1))

        st.subheader("🏆 Model Leaderboard")
        st.dataframe(results_df, use_container_width=True)

        best_model = results_df.iloc[0]
        st.success(f"Best Model: {best_model['Model']} ({best_model['R² Score']})")

        fig = px.bar(
            results_df,
            x="Model",
            y="R² Score",
            color="Model",
            text="R² Score",
            title="Model R² Comparison",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)