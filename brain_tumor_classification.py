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
from sklearn.tree import plot_tree
from matplotlib.colors import ListedColormap


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
    plt.figure(figsize=(6, 4))
    sns.countplot(x=class_column, data=data)
    plt.title("Class Distribution")
    plt.xlabel(f"{class_column} (0 = Non-Tumor, 1 = Tumor)")
    plt.ylabel("Count")
    plt.show()

    numeric_data = data.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()


def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(("blue", "orange")))
    plt.scatter(
        X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=ListedColormap(("blue", "orange"))
    )
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def plot_svm_margin(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=30)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50)
    )
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contour(
        xx,
        yy,
        Z,
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.7,
        linestyles=["--", "-", "--"],
    )
    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    plt.title("SVM Decision Boundary and Margin")
    plt.show()


def plot_decision_tree(model, feature_names):
    plt.figure(figsize=(12, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["Non-Tumor", "Tumor"],
        filled=True,
    )
    plt.title("Decision Tree Visualization")
    plt.show()


def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importance)), importance[indices], align="center")
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
    plt.title("Feature Importance in Random Forest")
    plt.show()


def plot_shap_values(model, X, feature_names):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names)


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
                "Accuracy": accuracy,
                "F1 Score": f1,
                "Precision": precision,
                "Recall": recall,
            }
        )

        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix for {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve for {name}")
            plt.legend(loc="lower right")
            plt.show()

        if name == "Decision Tree":
            plot_decision_tree(model, feature_names)

        if name == "Random Forest":
            plot_feature_importance(model, feature_names)

        if name == "Gradient Boosting":
            plot_shap_values(model, X_test, feature_names)

    return pd.DataFrame(results)


if __name__ == "__main__":
    filepath = r"archive\Brain Tumor.csv"
    X, y, data = load_and_preprocess_data(filepath)
    perform_eda(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, data.columns[:-1]
    )

    print("\nModel Performance Summary:")
    print(results)
