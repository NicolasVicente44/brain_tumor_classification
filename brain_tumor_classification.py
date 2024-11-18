# brain_tumor_basic.py

# 1. Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# 2. Load and Preprocess Data
def load_and_preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)

    # Separate features and target
    X = data.drop(
        columns=["Class", "Image"]
    )  # 'Class' is the label, 'Image' is just an identifier
    y = data["Class"]

    # Print the first few rows of X to inspect for any concatenated values
    print(X.head())

    # Attempt to convert all columns in X to numeric, forcing errors to NaN
    X = X.apply(pd.to_numeric, errors="coerce")

    # Print rows with NaN values to identify any issues
    print("Rows with NaN values:", X[X.isna().any(axis=1)])

    # Fill NaN values with 0, or use another strategy
    X = X.fillna(0)

    # Scale features for consistency
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


# 3. Train Model (Random Forest)
def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf


# 4. Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)


# 5. Main Script
if __name__ == "__main__":
    # Path to your dataset file
    filepath = "archive/Brain Tumor.csv"  # Update this to the actual path

    # Load and preprocess data
    X, y = load_and_preprocess_data(filepath)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Random Forest model
    rf_model = train_random_forest(X_train, y_train)

    # Evaluate the model
    evaluate_model(rf_model, X_test, y_test)
