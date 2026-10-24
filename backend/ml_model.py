"""
ML Model — handles training, evaluation, prediction, and feature importance.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score


class MLModel:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None
        self.feature_columns = []
        self.target_column = None
        self.label_encoders = {}
        self.target_encoder = None
        self.accuracy = 0.0
        self.task_type = "classification"

    def train(self, target_column: str) -> float:
        self.target_column = target_column
        df = self.df.dropna()

        # Encode categorical feature columns
        for col in df.columns:
            if col == target_column:
                continue
            if df[col].dtype == object:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le

        # Encode target if categorical
        if df[target_column].dtype == object:
            self.target_encoder = LabelEncoder()
            df[target_column] = self.target_encoder.fit_transform(df[target_column].astype(str))
            self.task_type = "classification"
        else:
            unique_vals = df[target_column].nunique()
            self.task_type = "classification" if unique_vals <= 20 else "regression"

        self.feature_columns = [c for c in df.columns if c != target_column]
        X = df[self.feature_columns]
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if self.task_type == "classification":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            self.accuracy = accuracy_score(y_test, y_pred)
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            self.accuracy = r2_score(y_test, y_pred)

        return self.accuracy

    def get_accuracy(self) -> float:
        return self.accuracy

    def get_feature_importance(self) -> dict:
        if self.model is None:
            return {"error": "Train model first"}
        importances = self.model.feature_importances_
        fi = dict(zip(self.feature_columns, [round(float(i), 4) for i in importances]))
        return dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    def predict(self, input_text: str):
        if self.model is None:
            return "Train model first"
        try:
            values = [v.strip() for v in input_text.split(",")]
            if len(values) != len(self.feature_columns):
                return (
                    f"Expected {len(self.feature_columns)} values: "
                    f"{self.feature_columns}"
                )
            row = {}
            for col, val in zip(self.feature_columns, values):
                if col in self.label_encoders:
                    val = self.label_encoders[col].transform([val])[0]
                else:
                    val = float(val)
                row[col] = val

            X = pd.DataFrame([row])
            pred = self.model.predict(X)[0]

            if self.target_encoder:
                pred = self.target_encoder.inverse_transform([int(pred)])[0]

            return str(pred)
        except Exception as e:
            return f"Prediction error: {str(e)}"