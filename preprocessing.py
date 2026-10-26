"""
preprocessing.py
-----------------
Builds the scikit-learn preprocessing pipeline: standard-scale the
numeric features, one-hot encode the categorical ones. Kept separate
from train_model.py so evaluate.py and predict.py can reuse the exact
same transform via the saved pipeline.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
        ]
    )


def clean_dataframe(df):
    """Basic cleaning: drop exact duplicates, drop rows missing the target."""
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=[config.TARGET_COL])
    return df
