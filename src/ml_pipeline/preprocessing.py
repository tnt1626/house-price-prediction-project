"""
Preprocessing and feature engineering module for the ML pipeline.
Contains all data transformers and feature engineering logic.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
import mlflow

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.core.config import (
    ORDINAL_MAP_CANONICAL,
    TARGET_ENCODER_COLS,
    TARGET_ENCODER_ALPHA,
    RARE_POOLER_MIN_COUNT,
    QUANTILE_N_QUANTILES,
    QUANTILE_SUBSAMPLE,
    SCALERS_DIR
)
from src.core.utils import Logger, log_to_mlflow


logger = Logger(__name__)


class OrdinalMapper(BaseEstimator, TransformerMixin):
    """Maps ordinal categorical variables to numeric values based on predefined order"""
    
    def __init__(self, mapping: Dict[str, List[str]]):
        """
        Initialize ordinal mapper.
        
        Parameters:
            mapping: Dictionary mapping feature names to ordered category lists
        """
        self.mapping = mapping
        self.maps_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Learn mappings from data"""
        self.maps_ = {}
        for col, order in self.mapping.items():
            if col in X.columns:
                self.maps_[col] = {v: i for i, v in enumerate(order)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply ordinal mapping"""
        X = X.copy()
        for col, m in self.maps_.items():
            if col in X.columns:
                X[col] = X[col].map(m)
        return X


class MissingnessIndicator(BaseEstimator, TransformerMixin):
    """Creates binary indicators for missing values in numeric columns"""
    
    def __init__(self, cols: Optional[List[str]] = None, auto_numeric: bool = True):
        """
        Initialize missingness indicator.
        
        Parameters:
            cols: Specific columns to track (if None, auto-detect)
            auto_numeric: Whether to auto-detect numeric columns with missing values
        """
        self.cols = cols
        self.auto_numeric = auto_numeric
        self.cols_ = []

    def fit(self, X: pd.DataFrame, y=None):
        """Identify columns with missing values"""
        if self.cols is not None:
            self.cols_ = [c for c in self.cols if c in X.columns]
        elif self.auto_numeric:
            num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
            self.cols_ = [c for c in num_cols if X[c].isna().any()]
        else:
            self.cols_ = []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add missingness indicator columns"""
        X = X.copy()
        for c in self.cols_:
            X[f"{c}_was_missing"] = X[c].isna().astype(int)
        return X


class RarePooler(BaseEstimator, TransformerMixin):
    """
    Pools rare categories into 'Other'.
    Supports both min_count (absolute frequency) and min_perc (percentage).
    """
    
    def __init__(
        self, 
        cols: List[str], 
        min_count: Optional[int] = None, 
        min_perc: Optional[float] = None
    ):
        """
        Initialize rare pooler.
        
        Parameters:
            cols: Columns to apply rare pooling to
            min_count: Minimum absolute frequency to keep
            min_perc: Minimum percentage frequency to keep (0-1)
        """
        self.cols = cols
        self.min_count = min_count
        self.min_perc = min_perc
        self.keep_levels_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        """Learn which category levels to keep"""
        self.keep_levels_ = {}
        n_samples = len(X)

        # Determine threshold
        threshold = self.min_count if self.min_count is not None else 0
        if self.min_perc is not None:
            threshold = max(threshold, self.min_perc * n_samples)

        for c in self.cols:
            if c in X.columns:
                vc = X[c].value_counts(dropna=False)
                self.keep_levels_[c] = set(vc[vc >= threshold].index.astype(str))
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Pool rare categories"""
        X = X.copy()
        for c, keep in self.keep_levels_.items():
            if c in X.columns:
                X[c] = X[c].astype(str)
                X[c] = np.where(X[c].isin(keep), X[c], "Other")
        return X


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Target-based encoding for categorical variables with smoothing"""
    
    def __init__(self, cols: Optional[List[str]] = None, alpha: float = 10.0):
        """
        Initialize target encoder.
        
        Parameters:
            cols: Columns to encode
            alpha: Smoothing factor (higher = more smoothing)
        """
        self.cols = cols or []
        self.alpha = alpha
        self.mapping_ = {}
        self.global_mean_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Learn target encoding mappings"""
        self.global_mean_ = y.mean()
        self.mapping_ = {}
        
        for col in self.cols:
            if col in X.columns:
                agg = y.groupby(X[col]).agg(['count', 'mean'])
                counts = agg['count']
                means = agg['mean']
                # Apply smoothing
                smooth = (counts * means + self.alpha * self.global_mean_) / (counts + self.alpha)
                self.mapping_[col] = smooth.to_dict()
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply target encoding"""
        X_out = X.copy()
        for col, map_dict in self.mapping_.items():
            X_out[f"TE_{col}"] = X_out[col].map(map_dict).fillna(self.global_mean_)
        return X_out


class DataSanitizer(BaseEstimator, TransformerMixin):
    """Handles infinite values and NaN for both DataFrame and NumPy arrays"""
    
    def __init__(self):
        self.keep_cols_idx_ = []
        self.is_df_ = False

    def fit(self, X, y=None):
        """Identify columns to keep"""
        if hasattr(X, 'select_dtypes'):
            self.keep_cols_idx_ = [i for i, col in enumerate(X.columns) 
                                   if not X[col].isnull().all()]
            self.is_df_ = True
        else:
            self.keep_cols_idx_ = [i for i in range(X.shape[1]) 
                                   if not np.all(np.isnan(X[:, i]))]
            self.is_df_ = False
        return self

    def transform(self, X):
        """Remove infinite values and empty columns"""
        if hasattr(X, 'values'):
            X_out = X.values.copy()
        else:
            X_out = X.copy()

        # Replace infinite values with NaN
        X_out = np.where(np.isinf(X_out), np.nan, X_out)

        # Keep only valid columns
        X_out = X_out[:, self.keep_cols_idx_]

        return X_out


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific real estate features.
    
    Parameters:
        df: Input DataFrame
        
    Returns:
        DataFrame with additional engineered features
    """
    df = df.copy()

    # 1. Total area and bathroom features
    df["TotalSF"] = df[["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]].fillna(0).sum(axis=1)
    df["TotalBath"] = (df["FullBath"].fillna(0) + 0.5*df["HalfBath"].fillna(0) +
                       df["BsmtFullBath"].fillna(0) + 0.5*df["BsmtHalfBath"].fillna(0))

    # 2. Age-related features
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    # 3. Binary and ratio features
    df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)
    df["TotalPorchSF"] = df[["OpenPorchSF", "EnclosedPorch", "3SsnPorch", 
                              "ScreenPorch", "WoodDeckSF"]].fillna(0).sum(axis=1)

    # 4. Quality-Area interaction
    if "OverallQual" in df.columns:
        df["Quality_Area_Interaction"] = df["OverallQual"] * df["GrLivArea"]

    # 5. Cyclic encoding for month sold
    if "MoSold" in df.columns:
        df["MoSold_sin"] = np.sin(2 * np.pi * df["MoSold"] / 12)
        df["MoSold_cos"] = np.cos(2 * np.pi * df["MoSold"] / 12)

    # 6. Location-type interaction
    if "Neighborhood" in df.columns and "BldgType" in df.columns:
        df["Loc_Type"] = df["Neighborhood"].astype(str) + "_" + df["BldgType"].astype(str)

    # 7. Outlier clipping
    if "LotArea" in df.columns:
        df["LotArea_clip"] = df["LotArea"].clip(upper=df["LotArea"].quantile(0.99))

    logger.info("[OK] Domain features added")
    return df


def build_feature_lists(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame
) -> tuple:
    """
    Categorize features into different types for preprocessing.
    
    Parameters:
        df_train: Training DataFrame
        df_test: Test DataFrame
        
    Returns:
        Tuple of (categorical, ordinal, numeric_continuous, numeric_absence)
    """
    # Ensure MSSubClass is treated as categorical
    for df in (df_train, df_test):
        if "MSSubClass" in df.columns:
            df["MSSubClass"] = df["MSSubClass"].astype(str)

    all_cols = df_train.drop(columns=["SalePrice"], errors="ignore").columns

    # Ordinal features
    ord_cols = [c for c in ORDINAL_MAP_CANONICAL.keys() if c in all_cols]
    
    # Categorical features
    cat_cols = [c for c in all_cols if (df_train[c].dtype == "object") and (c not in ord_cols)]
    
    # Numeric features
    num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df_train[c]) and c not in ord_cols]

    # Numeric with missing values
    num_abs_candidates = []
    for c in num_cols:
        if df_train[c].isna().any() or df_test[c].isna().any():
            num_abs_candidates.append(c)

    num_cont = [c for c in num_cols if c not in num_abs_candidates]
    
    logger.info(f"[OK] Feature categorization: {len(cat_cols)} categorical, "
                f"{len(ord_cols)} ordinal, {len(num_cont)} continuous, "
                f"{len(num_abs_candidates)} with missing")
    
    return cat_cols, ord_cols, num_cont, num_abs_candidates


def make_preprocessor(
    cat_cols: List[str],
    ord_cols: List[str],
    num_cont: List[str],
    num_absence: List[str]
) -> Pipeline:
    """
    Create preprocessing pipeline.
    
    Parameters:
        cat_cols: Categorical columns
        ord_cols: Ordinal columns
        num_cont: Continuous numeric columns
        num_absence: Numeric columns with missing values
        
    Returns:
        Configured preprocessing Pipeline
    """
    # Pre-processing steps
    pre_steps = [
        ("ordinal_map", OrdinalMapper(ORDINAL_MAP_CANONICAL)),
        ("missing_flags", MissingnessIndicator(cols=None, auto_numeric=True)),
        ("rare_pool", RarePooler(cat_cols, min_count=RARE_POOLER_MIN_COUNT)),
        ("te", TargetEncoder(cols=[c for c in TARGET_ENCODER_COLS if c in cat_cols], 
                            alpha=TARGET_ENCODER_ALPHA)),
    ]
    pre = Pipeline(steps=pre_steps)

    # Handle sklearn version differences
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    # Column transformers
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])

    ord_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])

    num_cont_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("qntl", QuantileTransformer(output_distribution="normal", 
                                     n_quantiles=QUANTILE_N_QUANTILES, 
                                     subsample=QUANTILE_SUBSAMPLE, copy=True)),
    ])

    num_abs_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])

    ct = ColumnTransformer(
        transformers=[
            ("cats", cat_pipe, cat_cols),
            ("ords", ord_pipe, ord_cols),
            ("num_cont", num_cont_pipe, num_cont),
            ("num_abs", num_abs_pipe, num_absence),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    return Pipeline([("prep", pre), ("ct", ct), ("sanitizer", DataSanitizer())])


def make_feature_space(
    df_train: pd.DataFrame, 
    df_test: Optional[pd.DataFrame] = None
) -> Pipeline:
    """
    Create complete feature engineering pipeline.
    
    Parameters:
        df_train: Training DataFrame
        df_test: Test DataFrame (uses training if None)
        
    Returns:
        Complete feature engineering Pipeline
    """
    if df_test is None:
        df_test = df_train
    
    df_train_aug = add_domain_features(df_train.copy())
    df_test_aug = add_domain_features(df_test.copy())
    cat_cols, ord_cols, num_cont, num_abs = build_feature_lists(df_train_aug, df_test_aug)

    return Pipeline([
        ("add_domain", FunctionTransformer(add_domain_features)),
        ("preproc", make_preprocessor(cat_cols, ord_cols, num_cont, num_abs)),
    ])


def save_scalers(preprocessor: Pipeline, output_dir: Path = SCALERS_DIR) -> bool:
    """
    Save trained transformers/scalers for inference.
    
    Parameters:
        preprocessor: Fitted preprocessing pipeline
        output_dir: Directory to save scalers
        
    Returns:
        True if successful
    """
    import joblib
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        scaler_path = output_dir / "preprocessor.joblib"
        joblib.dump(preprocessor, scaler_path)
        logger.info(f"[OK] Scalers saved to {scaler_path}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to save scalers: {e}")
        return False
