# ============================================================================
# Custom Transformers for Feature Engineering
# ============================================================================
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer as SkQuantileTransformer


class OrdinalMapper(BaseEstimator, TransformerMixin):
    """Maps ordinal categorical variables to numeric values based on predefined order"""
    def __init__(self, mapping):
        self.mapping = mapping
        self.maps_ = {}

    def fit(self, X, y=None):
        self.maps_ = {}
        for col, order in self.mapping.items():
            if col in X.columns:
                self.maps_[col] = {v: i for i, v in enumerate(order)}
        return self

    def transform(self, X):
        X = X.copy()
        for col, m in self.maps_.items():
            X[col] = X[col].map(m)
        return X


class MissingnessIndicator(BaseEstimator, TransformerMixin):
    """Creates binary indicators for missing values in numeric columns"""
    def __init__(self, cols=None, auto_numeric=True):
        self.cols = cols
        self.auto_numeric = auto_numeric
        self.cols_ = []

    def fit(self, X, y=None):
        if self.cols is not None:
            self.cols_ = [c for c in self.cols if c in X.columns]
        elif self.auto_numeric:
            num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
            self.cols_ = [c for c in num_cols if X[c].isna().any()]
        else:
            self.cols_ = []
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols_:
            X[f"{c}_was_missing"] = X[c].isna().astype(int)
        return X


class RarePooler(BaseEstimator, TransformerMixin):
    """Pools rare categories into 'Other'"""
    def __init__(self, cols, min_count=None, min_perc=None):
        self.cols = cols
        self.min_count = min_count
        self.min_perc = min_perc
        self.keep_levels_ = {}

    def fit(self, X, y=None):
        self.keep_levels_ = {}
        n_samples = len(X)

        threshold = self.min_count if self.min_count is not None else 0
        if self.min_perc is not None:
            threshold = max(threshold, self.min_perc * n_samples)

        for c in self.cols:
            if c in X.columns:
                vc = X[c].value_counts(dropna=False)
                self.keep_levels_[c] = set(vc[vc >= threshold].index.astype(str))
        return self

    def transform(self, X):
        X = X.copy()
        for c, keep in self.keep_levels_.items():
            if c in X.columns:
                X[c] = X[c].astype(str)
                X[c] = np.where(X[c].isin(keep), X[c], "Other")
        return X


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoding for categorical variables"""
    def __init__(self, cols=None, alpha=10):
        self.cols = cols if cols is not None else []
        self.alpha = alpha
        self.mapping_ = {}
        self.global_mean_ = None

    def fit(self, X, y):
        self.global_mean_ = y.mean()
        for col in self.cols:
            if col in X.columns:
                agg = y.groupby(X[col]).agg(['count', 'mean'])
                counts = agg['count']
                means = agg['mean']
                smooth = (counts * means + self.alpha * self.global_mean_) / (counts + self.alpha)
                self.mapping_[col] = smooth.to_dict()
        return self

    def transform(self, X):
        X_out = X.copy()
        for col, map_dict in self.mapping_.items():
            if col in X.columns:
                X_out[f"TE_{col}"] = X_out[col].map(map_dict).fillna(self.global_mean_)
        return X_out


class DataSanitizer(BaseEstimator, TransformerMixin):
    """Handles infinite and NaN values for both DataFrame and NumPy array"""
    def fit(self, X, y=None):
        if hasattr(X, 'select_dtypes'):
            self.keep_cols_idx_ = [i for i, col in enumerate(X.columns) if not X[col].isnull().all()]
            self.is_df_ = True
        else:
            self.keep_cols_idx_ = [i for i in range(X.shape[1]) if not np.all(np.isnan(X[:, i]))]
            self.is_df_ = False
        return self

    def transform(self, X):
        if hasattr(X, 'values'):
            X_out = X.values.copy()
        else:
            X_out = X.copy()

        # Handle infinite values
        X_out = np.where(np.isinf(X_out), np.nan, X_out)

        # Keep valid columns only
        X_out = X_out[:, self.keep_cols_idx_]

        return X_out
