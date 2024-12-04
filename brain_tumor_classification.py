import pandas as pd
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
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
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_test, y_proba, model_name):
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.title(f"ROC Curve for {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(y_test, y_proba, model_name):
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AP = {pr_auc:.2f}")
    plt.title(f"Precision-Recall Curve for {model_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid()
    plt.tight_layout()
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


def plot_calibration_curve(y_test, y_proba, model_name):
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=1, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")
    plt.title(f"Calibration Curve for {model_name}")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()
    plt.show()


def train_and_evaluate_models_with_visuals(
    X_train, X_test, y_train, y_test, feature_names, version_name
):
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=5),
        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = []
    roc_data = []
    pr_data = []

    for name, model in classifiers.items():
        print(f"\n[{version_name}] Training and Evaluating Model: {name}")

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        print(f"Cross-Validation Scores ({name}): {scores}")
        print(f"Mean CV Accuracy: {scores.mean():.4f}, Std Dev: {scores.std():.4f}")

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        if hasattr(model, "predict_proba"):
            y_train_proba = model.predict_proba(X_train)[:, 1]
        else:
            y_train_scores = model.decision_function(X_train)
            y_train_proba = (y_train_scores - y_train_scores.min()) / (
                y_train_scores.max() - y_train_scores.min()
            )

        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_roc_auc = roc_auc_score(y_train, y_train_proba)

        y_test_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_test_scores = model.decision_function(X_test)
            y_test_proba = (y_test_scores - y_test_scores.min()) / (
                y_test_scores.max() - y_test_scores.min()
            )

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_proba)
        test_pr_auc = average_precision_score(y_test, y_test_proba)

        print(
            f"Training Accuracy: {train_accuracy:.4f}, Testing Accuracy: {test_accuracy:.4f}"
        )
        print(f"Training F1 Score: {train_f1:.4f}, Testing F1 Score: {test_f1:.4f}")
        print(
            f"Training ROC AUC: {train_roc_auc:.4f}, Testing ROC AUC: {test_roc_auc:.4f}"
        )

        results.append(
            {
                "Model": name,
                "CV Accuracy": scores.mean(),
                "Train Accuracy": train_accuracy,
                "Train F1 Score": train_f1,
                "Test Accuracy": test_accuracy,
                "Test F1 Score": test_f1,
                "Train Precision": train_precision,
                "Test Precision": test_precision,
                "Train Recall": train_recall,
                "Test Recall": test_recall,
                "Train ROC AUC": train_roc_auc,
                "Test ROC AUC": test_roc_auc,
                "Test PR AUC": test_pr_auc,
            }
        )

        confusion = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues")
        plt.title(f"[{version_name}] Confusion Matrix for {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        roc_data.append((name, fpr, tpr, test_roc_auc))
        plot_roc_curve(y_test, y_test_proba, f"{name} ({version_name})")

        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_proba)
        pr_data.append((name, precision_vals, recall_vals, test_pr_auc))
        plot_precision_recall_curve(y_test, y_test_proba, f"{name} ({version_name})")

        plot_calibration_curve(y_test, y_test_proba, f"{name} ({version_name})")

        if hasattr(model, "feature_importances_"):
            print(f"\n[{version_name}] Feature Importance for {name}:")
            plot_feature_importance(model, feature_names, f"{name} ({version_name})")

    if roc_data:
        plt.figure(figsize=(8, 6))
        for model_name, fpr, tpr, roc_auc in roc_data:
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
        plt.title(f"Combined ROC Curves ({version_name})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid()
        plt.tight_layout()
        plt.show()

    if pr_data:
        plt.figure(figsize=(8, 6))
        for model_name, precision_vals, recall_vals, pr_auc in pr_data:
            plt.plot(
                recall_vals, precision_vals, label=f"{model_name} (AP = {pr_auc:.2f})"
            )
        plt.title(f"Combined Precision-Recall Curves ({version_name})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.grid()
        plt.tight_layout()
        plt.show()

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
