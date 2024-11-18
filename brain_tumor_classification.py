import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import re

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)

    def clean_numeric(value):
        if isinstance(value, str):
            if re.match(r"^\d+(\.\d+)?\d+(\.\d+)?$", value):
                return float(value.split(".")[0])
            try:
                return float(value)
            except ValueError:
                return np.nan
        return value

    for column in data.columns.drop(["Class", "Image"]):
        data[column] = data[column].apply(clean_numeric)

    X = data.drop(columns=["Class", "Image"])
    y = data["Class"]

    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def evaluate_model(rf_model, X_test, y_test):
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

if __name__ == "__main__":
    filepath = r"archive\Brain Tumor.csv"

    X, y = load_and_preprocess_data(filepath)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    rf_model = train_random_forest(X_train, y_train)

    evaluate_model(rf_model, X_test, y_test)
