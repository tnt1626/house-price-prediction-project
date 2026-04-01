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
        if not isinstance(X, pd.DataFrame):
            return X
            
        X = X.copy()
        for col, m in self.maps_.items():
            if col in X.columns:
                X[col] = X[col].map(m)
        return X
    
    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else None


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
        if not isinstance(X, pd.DataFrame):
            return X
            
        X = X.copy()
        for c in self.cols_:
            # KIỂM TRA: Chỉ xử lý nếu cột tồn tại trong DataFrame hiện tại
            if c in X.columns:
                X[f"{c}_was_missing"] = X[c].isna().astype(int)
        return X
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None: return None
        # Trả về danh sách cột cũ cộng với các cột indicator mới
        new_cols = [f"{c}_was_missing" for c in self.cols_]
        return np.concatenate([input_features, new_cols])


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
        if not isinstance(X, pd.DataFrame):
            return X
        X = X.copy()
        for c, keep in self.keep_levels_.items():
            if c in X.columns:
                X[c] = X[c].astype(str)
                X[c] = np.where(X[c].isin(keep), X[c], "Other")
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else None

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
        if not isinstance(X, pd.DataFrame):
            return X
        X_out = X.copy()
        for col, map_dict in self.mapping_.items():
            X_out[f"TE_{col}"] = X_out[col].map(map_dict).fillna(self.global_mean_)
        return X_out
    
    def get_feature_names_out(self, input_features=None):
        return np.array(input_features) if input_features is not None else None


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
        if isinstance(X, pd.DataFrame):
            X_out = X.copy()
            X_out = X_out.replace([np.inf, -np.inf], np.nan)
            return X_out.iloc[:, self.keep_cols_idx_]

        X_out = np.where(np.isinf(X), np.nan, X)
        return X_out[:, self.keep_cols_idx_]
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None: return None
        # Trả về danh sách cột đã ~~được giữ lại (lọc theo keep_cols_idx_)
        return np.array(input_features)[self.keep_cols_idx_]


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific real estate features.
    Handles both schema names and actual data column names.
    Gracefully handles missing columns by using sensible defaults.
    
    Parameters:
        df: Input DataFrame
        
    Returns:
        DataFrame with additional engineered features
    """
    from src.core.config import SCHEMA_TO_DATA_MAPPING

    if not isinstance(df, pd.DataFrame):
        return df
    
    pd.set_option('future.no_silent_downcasting', True)
    
    df = df.copy()
    
    has_numeric_feature_cols = any(col.startswith('f') and col[1:].isdigit() for col in df.columns if len(col) > 1)
    if has_numeric_feature_cols and len(df.columns) > 100:
        logger.debug("[DEBUG] Skipping add_domain_features - data already transformed")
        return df
    
    rename_map = {}
    for schema_name, data_name in SCHEMA_TO_DATA_MAPPING.items():
        if schema_name in df.columns and data_name not in df.columns:
            rename_map[schema_name] = data_name
    
    if rename_map:
        logger.debug(f"[DEBUG] Renaming columns: {rename_map}")
        df = df.rename(columns=rename_map)
    
    required_numeric_cols = {
        'YrSold': 0,
        'YearBuilt': 0,
        'YearRemodAdd': 0,
        'OverallQual': 5,
        'GrLivArea': 0,
        'LotArea': 0
    }
    for col, default_val in required_numeric_cols.items():
        if col not in df.columns:
            df[col] = default_val

    floor_cols = []
    for col_variant in [["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"],
                       ["TotalBsmtSF", "FirstFlrSF", "SecondFlrSF"]]:
        if all(c in df.columns for c in col_variant):
            floor_cols = col_variant
            break
    
    if floor_cols:
        df["TotalSF"] = df[floor_cols].fillna(0).sum(axis=1)
    else:
        logger.debug(f"[DEBUG] Floor columns not found, creating TotalSF with 0")
        df["TotalSF"] = 0
    
    bath_cols = {
        "full_bath": None,
        "half_bath": None,
        "bsmt_full_bath": None,
        "bsmt_half_bath": None
    }

    for col in df.columns:
        col_lower = col.lower().replace(" ", "").replace("_", "")
        if "fullbath" in col_lower and "bsmt" not in col_lower and bath_cols["full_bath"] is None:
            bath_cols["full_bath"] = col
        elif "halfbath" in col_lower and "bsmt" not in col_lower and bath_cols["half_bath"] is None:
            bath_cols["half_bath"] = col
        elif "bsmtfullbath" in col_lower and bath_cols["bsmt_full_bath"] is None:
            bath_cols["bsmt_full_bath"] = col
        elif "bsmthalfbath" in col_lower and bath_cols["bsmt_half_bath"] is None:
            bath_cols["bsmt_half_bath"] = col
    
    try:
        total_bath = pd.Series(0.0, index=df.index)
        for key, col in bath_cols.items():
            if col and col in df.columns:
                if "half" in key:
                    total_bath += 0.5 * df[col].fillna(0)
                else:
                    total_bath += df[col].fillna(0)
        df["TotalBath"] = total_bath
        if total_bath.sum() == 0:
            logger.debug("[DEBUG] No bathroom data found, TotalBath set to 0")
    except Exception as e:
        logger.warning(f"Could not create TotalBath feature: {e}")
        df["TotalBath"] = 0

    if "YrSold" in df.columns and "YearBuilt" in df.columns:
        df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    else:
        df["HouseAge"] = 0
        
    if "YrSold" in df.columns and "YearRemodAdd" in df.columns:
        df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    else:
        df["RemodAge"] = 0
        
    if "YearRemodAdd" in df.columns and "YearBuilt" in df.columns:
        df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    else:
        df["IsRemodeled"] = 0

    second_flr_col = None
    for col in df.columns:
        col_lower = col.lower()
        if ("secondflr" in col_lower or "2ndflr" in col_lower) and "sf" in col_lower:
            second_flr_col = col
            break
    
    if second_flr_col and second_flr_col in df.columns:
        df["Has2ndFlr"] = (df[second_flr_col] > 0).astype(int)
    else:
        df["Has2ndFlr"] = 0
    
    porch_cols = [c for c in ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", 
                              "ScreenPorch", "WoodDeckSF"] if c in df.columns]
    if porch_cols:
        df["TotalPorchSF"] = df[porch_cols].fillna(0).sum(axis=1)
    else:
        df["TotalPorchSF"] = 0

    if "OverallQual" in df.columns and "GrLivArea" in df.columns:
        df["Quality_Area_Interaction"] = df["OverallQual"] * df["GrLivArea"]
    else:
        df["Quality_Area_Interaction"] = 0

    if "MoSold" in df.columns:
        try:
            # Ensure MoSold is numeric before applying trigonometric functions
            mo_sold_values = pd.to_numeric(df["MoSold"].fillna(6), errors='coerce')
            mo_sold_values = np.asarray(mo_sold_values, dtype=np.float64)
            df["MoSold_sin"] = np.sin(2 * np.pi * (mo_sold_values / 12))
            df["MoSold_cos"] = np.cos(2 * np.pi * (mo_sold_values / 12))
        except Exception as e:
            logger.warning(f"Could not create MoSold trigonometric features: {e}")
            df["MoSold_sin"] = 0
            df["MoSold_cos"] = 0
    else:
        df["MoSold_sin"] = 0
        df["MoSold_cos"] = 0

    if "Neighborhood" in df.columns and "BldgType" in df.columns:
        df["Loc_Type"] = df["Neighborhood"].astype(str) + "_" + df["BldgType"].astype(str)
    else:
        df["Loc_Type"] = "Unknown_Unknown"

    if "LotArea" in df.columns:
        lot_area_q99 = df["LotArea"].quantile(0.99)
        if pd.notna(lot_area_q99):
            df["LotArea_clip"] = df["LotArea"].clip(upper=lot_area_q99)
        else:
            df["LotArea_clip"] = df["LotArea"].fillna(0)
    else:
        df["LotArea_clip"] = 0

    logger.info("[OK] Domain features added")
    return df


def build_feature_lists(
    df_train: pd.DataFrame, 
    df_test: pd.DataFrame
) -> tuple:
    """
    Categorize features into different types for preprocessing.
    Handles both schema names and actual data column names.
    
    Parameters:
        df_train: Training DataFrame
        df_test: Test DataFrame
        
    Returns:
        Tuple of (categorical, ordinal, numeric_continuous, numeric_absence)
    """
    from src.core.config import SCHEMA_TO_DATA_MAPPING
    
    for schema_name, data_name in SCHEMA_TO_DATA_MAPPING.items():
        for df in (df_train, df_test):
            if schema_name in df.columns and data_name not in df.columns:
                df.rename(columns={schema_name: data_name}, inplace=True)
    
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
    logger.debug(f"[DEBUG] Numeric columns: {num_cols}")
    logger.debug(f"[DEBUG] Categorical columns: {cat_cols}")
    
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

    pipeline = Pipeline([
        ("add_domain", FunctionTransformer(add_domain_features)),
        ("preproc", make_preprocessor(cat_cols, ord_cols, num_cont, num_abs)),
    ])
    
    return pipeline

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
