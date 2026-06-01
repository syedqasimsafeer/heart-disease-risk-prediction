
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

RANDOM_STATE = 42

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

@st.cache_data
def load_data():
    data = pd.read_csv("data/heart_disease_clean.csv")
    return data

@st.cache_resource
def train_models(data):
    X = data.drop("target", axis=1)
    y = data["target"]

    numeric_features = [
        "age",
        "resting_blood_pressure",
        "cholesterol",
        "max_heart_rate",
        "st_depression"
    ]

    categorical_features = [
        "sex",
        "chest_pain_type",
        "fasting_blood_sugar",
        "resting_ecg",
        "exercise_angina",
        "st_slope",
        "major_vessels",
        "thalassemia"
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        )
    }

    results = []
    trained_models = {}

    for model_name, classifier in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        results.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_proba)
        })

        trained_models[model_name] = pipeline

    results_df = pd.DataFrame(results).sort_values(by="ROC AUC", ascending=False)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    return {
        "X_test": X_test,
        "y_test": y_test,
        "results_df": results_df,
        "trained_models": trained_models,
        "best_model_name": best_model_name,
        "best_model": best_model
    }

def create_input_dataframe():
    st.sidebar.header("Enter Patient Information")

    age = st.sidebar.slider("Age", 20, 90, 50)
    sex_label = st.sidebar.selectbox("Sex", ["Female", "Male"])
    sex = 1 if sex_label == "Male" else 0

    chest_pain_type = st.sidebar.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3],
        help="Encoded medical category from the dataset"
    )

    resting_blood_pressure = st.sidebar.slider(
        "Resting Blood Pressure",
        80,
        220,
        120
    )

    cholesterol = st.sidebar.slider(
        "Cholesterol",
        100,
        600,
        240
    )

    fasting_blood_sugar_label = st.sidebar.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["No", "Yes"]
    )
    fasting_blood_sugar = 1 if fasting_blood_sugar_label == "Yes" else 0

    resting_ecg = st.sidebar.selectbox("Resting ECG", [0, 1, 2])

    max_heart_rate = st.sidebar.slider(
        "Maximum Heart Rate",
        60,
        220,
        150
    )

    exercise_angina_label = st.sidebar.selectbox(
        "Exercise Induced Angina",
        ["No", "Yes"]
    )
    exercise_angina = 1 if exercise_angina_label == "Yes" else 0

    st_depression = st.sidebar.slider(
        "ST Depression",
        0.0,
        7.0,
        1.0,
        step=0.1
    )

    st_slope = st.sidebar.selectbox("ST Slope", [0, 1, 2])
    major_vessels = st.sidebar.selectbox("Major Vessels", [0, 1, 2, 3, 4])
    thalassemia = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "chest_pain_type": [chest_pain_type],
        "resting_blood_pressure": [resting_blood_pressure],
        "cholesterol": [cholesterol],
        "fasting_blood_sugar": [fasting_blood_sugar],
        "resting_ecg": [resting_ecg],
        "max_heart_rate": [max_heart_rate],
        "exercise_angina": [exercise_angina],
        "st_depression": [st_depression],
        "st_slope": [st_slope],
        "major_vessels": [major_vessels],
        "thalassemia": [thalassemia]
    })

    return input_data

def main():
    data = load_data()
    model_bundle = train_models(data)

    st.title("❤️ Heart Disease Risk Prediction")
    st.write(
        "A complete educational machine learning classification project with model comparison, "
        "evaluation metrics, ROC curves, and deployment."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(data):,}")
    col2.metric("Features", data.shape[1] - 1)
    col3.metric("Best Model", model_bundle["best_model_name"])
    col4.metric("Best ROC-AUC", f"{model_bundle['results_df'].iloc[0]['ROC AUC']:.3f}")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Predict Risk",
        "📊 Model Dashboard",
        "📈 ROC Curves",
        "📄 Dataset"
    ])

    with tab1:
        left, right = st.columns([1, 1])

        with left:
            st.subheader("Patient Input")
            input_data = create_input_dataframe()
            st.dataframe(input_data, use_container_width=True)

        with right:
            st.subheader("Prediction Result")

            best_model = model_bundle["best_model"]
            prediction = best_model.predict(input_data)[0]
            probability = best_model.predict_proba(input_data)[0][1]

            fig = px.pie(
                names=["No Disease Probability", "Disease Probability"],
                values=[1 - probability, probability],
                title="Predicted Probability"
            )
            st.plotly_chart(fig, use_container_width=True)

            if prediction == 1:
                st.error(f"Predicted Class: Heart Disease Risk Detected")
            else:
                st.success(f"Predicted Class: No Heart Disease Risk Detected")

            st.metric("Disease Probability", f"{probability * 100:.2f}%")

            if probability < 0.35:
                st.info("Risk Category: Low")
            elif probability < 0.65:
                st.warning("Risk Category: Medium")
            else:
                st.error("Risk Category: High")

            st.warning(
                "This app is for educational purposes only. It is not a medical diagnosis tool."
            )

    with tab2:
        st.subheader("Model Comparison")

        results_df = model_bundle["results_df"]
        st.dataframe(results_df, use_container_width=True)

        fig = px.bar(
            results_df,
            x="Model",
            y=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
            barmode="group",
            title="Classification Metrics Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

        best_model = model_bundle["best_model"]
        X_test = model_bundle["X_test"]
        y_test = model_bundle["y_test"]

        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        cm_df = pd.DataFrame(
            cm,
            index=["Actual No Disease", "Actual Disease"],
            columns=["Predicted No Disease", "Predicted Disease"]
        )

        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            title=f"Confusion Matrix - {model_bundle['best_model_name']}",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with tab3:
        st.subheader("ROC Curve Comparison")

        X_test = model_bundle["X_test"]
        y_test = model_bundle["y_test"]

        roc_data = []

        for model_name, model in model_bundle["trained_models"].items():
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            auc_score = roc_auc_score(y_test, y_proba)

            for fp, tp in zip(fpr, tpr):
                roc_data.append({
                    "False Positive Rate": fp,
                    "True Positive Rate": tp,
                    "Model": f"{model_name} AUC={auc_score:.3f}"
                })

        roc_df = pd.DataFrame(roc_data)

        fig_roc = px.line(
            roc_df,
            x="False Positive Rate",
            y="True Positive Rate",
            color="Model",
            title="ROC Curves"
        )

        st.plotly_chart(fig_roc, use_container_width=True)

    with tab4:
        st.subheader("Dataset Preview")
        st.dataframe(data.head(50), use_container_width=True)

        st.subheader("Target Distribution")
        target_counts = data["target"].value_counts().reset_index()
        target_counts.columns = ["Target", "Count"]
        target_counts["Target"] = target_counts["Target"].map({
            0: "No Disease",
            1: "Disease"
        })

        fig_target = px.bar(
            target_counts,
            x="Target",
            y="Count",
            title="Target Class Distribution"
        )
        st.plotly_chart(fig_target, use_container_width=True)

        st.subheader("Correlation Heatmap")
        corr = data.corr(numeric_only=True)
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            title="Correlation Heatmap"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

if __name__ == "__main__":
    main()
