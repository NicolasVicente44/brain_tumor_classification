import pandas as pd
import numpy as np
import re
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)

    def clean_numeric(value):
        
        if isinstance(value, str):
            matches = re.findall(r"\d+(\.\d+)?", value)
            if matches:
                return float(matches[0])
            try:
                return float(value)
            except ValueError:
                return np.nan
        return value

    for column in data.columns.drop(["Class", "Image"]):
        data[column] = data[column].apply(clean_numeric)

    data = data.dropna()

    X = data.drop(columns=["Class", "Image"])
    y = data["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, data


def perform_eda(data):
    print("Columns in data:", data.columns)
    class_column = "Class"

    class_counts = data[class_column].value_counts()
    labels = [
        (
            f"Non-Tumor ({class_counts[0]} - {class_counts[0] / class_counts.sum() * 100:.2f}%)"
            if label == 0
            else f"Tumor ({class_counts[1]} - {class_counts[1] / class_counts.sum() * 100:.2f}%)"
        )
        for label in class_counts.index
    ]

    plt.figure(figsize=(10, 10))
    plt.pie(
        class_counts,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#ff9999", "#66b3ff"],
        labeldistance=1.1,
    )
    plt.title("Class Distribution (Counts and Percentages)")
    plt.tight_layout()
    plt.show()

    numeric_data = data.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()


def engineer_features(data):
    data["Mean_Variance_Ratio"] = data["Mean"] / (data["Variance"] + 1e-5)
    data["Energy_Entropy_Product"] = data["Energy"] * data["Entropy"]
    data["Skewness_Kurtosis_Sum"] = data["Skewness"] + data["Kurtosis"]
    data["Dissimilarity_Normalized"] = data["Dissimilarity"] / (data["ASM"] + 1e-5)
    data["Contrast_Homogeneity_Diff"] = data["Contrast"] - data["Homogeneity"]
    data["Entropy_Log"] = np.log(data["Entropy"] + 1e-5)
    return data


def plot_feature_importance(model, feature_names, model_name):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance[indices], align="center")
    plt.xticks(
        range(len(importance)),
        [feature_names[i] for i in indices],
        rotation=45,
        ha="right",
    )
    plt.title(f"Feature Importance ({model_name})")
    plt.tight_layout()
    plt.show()


def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names):
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = []

    for name, model in classifiers.items():
        print(f"\nTraining and Evaluating Model: {name}")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(
            f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )

        confusion = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(confusion)

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "F1 Score": f1,
                "Precision": precision,
                "Recall": recall,
            }
        )

        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix for {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = roc_auc_score(y_test, y_proba)
            print(f"ROC AUC Score: {roc_auc:.4f}")
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve for {name}")
            plt.legend(loc="lower right")
            plt.show()

        if name in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
            plot_feature_importance(model, feature_names, name)

    return pd.DataFrame(results)


def perform_analysis_with_engineered_features(filepath):
    data = pd.read_csv(filepath)

    def clean_numeric(value):
        if isinstance(value, str):
            matches = re.findall(r"\d+(\.\d+)?", value)
            if matches:
                return float(matches[0])
            try:
                return float(value)
            except ValueError:
                return np.nan
        return value

    for column in data.columns.drop(["Class", "Image"]):
        data[column] = data[column].apply(clean_numeric)

    data = data.dropna()

    data = engineer_features(data)

    X = data.drop(columns=["Class", "Image"])
    y = data["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    print("\nCross-Validation Scores for All Models (Engineered Features):")
    for name, model in classifiers.items():
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
        print(f"{name}:")
        print(f"Scores: {scores}")
        print(f"Mean Accuracy: {scores.mean():.4f}\n")

    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, X.columns)

    print("\nModel Performance Summary (Engineered Features):")
    print(results.to_string(index=False))


if __name__ == "__main__":
    filepath = r"archive\Brain Tumor.csv"

    print("\n==== Original Analysis ====")
    X, y, data = load_and_preprocess_data(filepath)
    perform_eda(data)

    print("\n==== Analysis with Engineered Features ====")
    perform_analysis_with_engineered_features(filepath)
