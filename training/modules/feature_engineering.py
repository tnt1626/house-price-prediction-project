# ============================================================================
# Feature Engineering Functions
# ============================================================================
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone

from .config import ORDINAL_MAP_CANONICAL, TARGET_ENCODER_FEATURES, TARGET_ENCODER_ALPHA
from .config import RARE_POOLER_MIN_COUNT, QUANTILE_TRANSFORMER_N_QUANTILES
from .transformers import OrdinalMapper, MissingnessIndicator, RarePooler, TargetEncoder, DataSanitizer


def add_domain_features(df):
    """Add domain-specific features for house price prediction"""
    df = df.copy()

    # 1. Total area & bathrooms
    df["TotalSF"] = df[["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"]].fillna(0).sum(axis=1)
    df["TotalBath"] = (df["FullBath"].fillna(0) + 0.5*df["HalfBath"].fillna(0) +
                       df["BsmtFullBath"].fillna(0) + 0.5*df["BsmtHalfBath"].fillna(0))

    # 2. House age features
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    # 3. Binary features
    df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)
    df["TotalPorchSF"] = df[["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]].fillna(0).sum(axis=1)

    # 4. Quality-Area interaction
    if "OverallQual" in df.columns:
        df["Quality_Area_Interaction"] = df["OverallQual"] * df["GrLivArea"]

    # 5. Cyclical features (Month of sale)
    if "MoSold" in df.columns:
        df["MoSold_sin"] = np.sin(2 * np.pi * df["MoSold"] / 12)
        df["MoSold_cos"] = np.cos(2 * np.pi * df["MoSold"] / 12)

    # 6. Interactions
    if "Neighborhood" in df.columns and "BldgType" in df.columns:
        df["Neighborhood_BldgType"] = df["Neighborhood"].astype(str) + "_" + df["BldgType"].astype(str)

    # 7. Clipping outliers
    if "LotArea" in df.columns:
        df["LotArea_clip"] = df["LotArea"].clip(upper=df["LotArea"].quantile(0.99))

    return df


def build_feature_lists(df_train, df_test):
    """Build feature lists for different types"""
    for df in (df_train, df_test):
        if "MSSubClass" in df.columns:
            df["MSSubClass"] = df["MSSubClass"].astype(str)

    all_cols = df_train.drop(columns=["SalePrice"], errors="ignore").columns

    ord_cols = [c for c in ORDINAL_MAP_CANONICAL.keys() if c in all_cols]
    cat_cols = [c for c in all_cols if (df_train[c].dtype == "object") and (c not in ord_cols)]
    num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df_train[c]) and c not in ord_cols]

    num_abs_candidates = []
    for c in num_cols:
        if df_train[c].isna().any() or df_test[c].isna().any():
            num_abs_candidates.append(c)

    num_cont = [c for c in num_cols if c not in num_abs_candidates]
    return cat_cols, ord_cols, num_cont, num_abs_candidates


def make_preprocessor(cat_cols, ord_cols, num_cont, num_absence):
    """Create the preprocessing pipeline"""
    
    te_cols = [c for c in TARGET_ENCODER_FEATURES if c in cat_cols]

    pre_steps = [
        ("ordinal_map", OrdinalMapper(ORDINAL_MAP_CANONICAL)),
        ("missing_flags", MissingnessIndicator(cols=None, auto_numeric=True)),
        ("rare_pool", RarePooler(cat_cols, min_count=RARE_POOLER_MIN_COUNT)),
        ("te", TargetEncoder(cols=te_cols, alpha=TARGET_ENCODER_ALPHA)),
    ]
    pre = Pipeline(steps=pre_steps)

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

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
                                    n_quantiles=QUANTILE_TRANSFORMER_N_QUANTILES, 
                                    subsample=200000, copy=True)),
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


def make_feature_space(df_train, df_test=None):
    """Create the complete feature engineering pipeline"""
    if df_test is None:
        df_test = df_train
        
    df_train_aug = add_domain_features(df_train.copy())
    df_test_aug = add_domain_features(df_test.copy())
    cat_cols, ord_cols, num_cont, num_abs = build_feature_lists(df_train_aug, df_test_aug)

    return Pipeline([
        ("add_domain", FunctionTransformer(add_domain_features)),
        ("preproc", make_preprocessor(cat_cols, ord_cols, num_cont, num_abs)),
    ])
