import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
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


def load_and_preprocess_original_features(filepath):
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


def engineer_features(data):
    engineered_features = {
        "Mean_Variance_Ratio": data["Mean"] / (data["Variance"] + 1e-5),
        "Energy_Entropy_Product": data["Energy"] * data["Entropy"],
        "Skewness_Kurtosis_Sum": data["Skewness"] + data["Kurtosis"],
        "Dissimilarity_Normalized": data["Dissimilarity"] / (data["ASM"] + 1e-5),
        "Contrast_Homogeneity_Diff": data["Contrast"] - data["Homogeneity"],
        "Entropy_Log": np.log(data["Entropy"] + 1e-5),
        "Intensity_Variability_Spread": (data["Variance"] * data["Entropy"])
        / (data["Mean"] + 1e-5),
        "Texture_Edge_Complexity": (data["Contrast"] * (1 - data["Homogeneity"]))
        / (data["Coarseness"] + 1e-5),
    }
    for name, values in engineered_features.items():
        data[name] = values
    return list(engineered_features.keys()), data


def load_and_preprocess_engineered_features(filepath):
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

    engineered_feature_names, data = engineer_features(data)

    X = data[engineered_feature_names]
    y = data["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, data, engineered_feature_names


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


def plot_roc_curve(y_test, y_proba, model_name):
    if y_proba is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.title(f"ROC Curve for {model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid()
        plt.show()


def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
        plt.title(f"Feature Importances for {model_name}")
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.show()


def train_and_evaluate_models_with_visuals(
    X_train, X_test, y_train, y_test, feature_names, version_name
):
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
        print(f"\n[{version_name}] Training and Evaluating Model: {name}")

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        print(f"Cross-Validation Scores ({name}): {scores}")
        print(f"Mean CV Accuracy: {scores.mean():.4f}, Std Dev: {scores.std():.4f}")

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

        results.append(
            {
                "Model": name,
                "CV Accuracy": scores.mean(),
                "Accuracy": accuracy,
                "F1 Score": f1,
                "Precision": precision,
                "Recall": recall,
            }
        )

        print(
            f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
        )

        confusion = confusion_matrix(y_test, y_pred)
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
        plt.title(f"[{version_name}] Confusion Matrix for {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        if y_proba is not None:
            plot_roc_curve(y_test, y_proba, f"{name} ({version_name})")

        if hasattr(model, "feature_importances_"):
            print(f"\n[{version_name}] Feature Importance for {name}:")
            plot_feature_importance(model, feature_names, f"{name} ({version_name})")

    return pd.DataFrame(results)


def train_and_compare(filepath):
    X_original, y_original, original_data = load_and_preprocess_original_features(
        filepath
    )
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        X_original, y_original, test_size=0.3, random_state=42
    )

    print("\n==== Training with Original Features ====")
    results_original = train_and_evaluate_models_with_visuals(
        X_train_orig,
        X_test_orig,
        y_train_orig,
        y_test_orig,
        original_data.columns.drop(["Class", "Image"]),
        version_name="Original Features",
    )

 
    X_engineered, y_engineered, engineered_data, engineered_feature_names = (
        load_and_preprocess_engineered_features(filepath)
    )
    X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(
        X_engineered, y_engineered, test_size=0.3, random_state=42
    )

    print("\n==== Training with Engineered Features ====")
    results_engineered = train_and_evaluate_models_with_visuals(
        X_train_eng,
        X_test_eng,
        y_train_eng,
        y_test_eng,
        engineered_feature_names, 
        version_name="Engineered Features",
    )

    return results_original, results_engineered


if __name__ == "__main__":
    filepath = r"archive\Brain Tumor.csv"

    X, y, data = load_and_preprocess_original_features(filepath)
    perform_eda(data)

    results_original, results_engineered = train_and_compare(filepath)

    print("\n==== Model Performance Comparison ====")
    print("Original Features Results:")
    print(results_original)
    print("\nEngineered Features Results:")
    print(results_engineered)
