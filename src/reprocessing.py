import pandas as pd

def make_preprocessor(cat_cols, ord_cols, num_cont, num_absence):
    """Create the preprocessing pipeline"""
    TE_DEFAULT = ["Neighborhood","MSZoning","Exterior1st","Exterior2nd","SaleCondition","BldgType","Neighborhood_BldgType"]

    pre_steps = [
        ("ordinal_map", OrdinalMapper(ORDINAL_MAP_CANONICAL)),
        ("missing_flags", MissingnessIndicator(cols=None, auto_numeric=True)),
        ("rare_pool", RarePooler(cat_cols, min_count=15)),
        ("te", TargetEncoder(cols=[c for c in TE_DEFAULT if c in cat_cols], alpha=30.0)),
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
        ("qntl", QuantileTransformer(output_distribution="normal", n_quantiles=200, subsample=200000, copy=True)),
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