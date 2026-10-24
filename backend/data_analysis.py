import pandas as pd
import numpy as np

class DataAnalyzer:

    def __init__(self, df):
        self.df = df

    def dataset_info(self):
        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "column_names": self.df.columns.tolist()
        }

    def missing_values(self):
        return self.df.isnull().sum().to_dict()

    def summary_statistics(self):
        return self.df.describe(include="all").fillna("").to_dict()

    def duplicate_rows(self):
        return int(self.df.duplicated().sum())

    def data_types(self):
        return {
            col: str(dtype)
            for col, dtype in self.df.dtypes.items()
        }

    def cleaning_suggestions(self):

        suggestions = []

        if self.df.isnull().sum().sum() > 0:
            suggestions.append(
                "Dataset contains missing values."
            )

        if self.df.duplicated().sum() > 0:
            suggestions.append(
                "Dataset contains duplicate rows."
            )

        empty_cols = [
            col for col in self.df.columns
            if self.df[col].isnull().sum() > 0.9 * len(self.df)
        ]

        if empty_cols:
            suggestions.append(
                f"Remove empty columns: {empty_cols}"
            )

        object_cols = self.df.select_dtypes(
            include="object"
        ).columns

        if len(object_cols) > 0:
            suggestions.append(
                "Categorical columns should be encoded."
            )

        return suggestions