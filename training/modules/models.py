# ============================================================================
# Model Definitions and Training Functions
# ============================================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from joblib import Parallel, delayed
import traceback

from .config import RANDOM_STATE, MODEL_PARAMS

# Optional model libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def get_scorers():
    """Get scorers for cross-validation"""
    return {
        'neg_rmse': 'neg_root_mean_squared_error',
        'r2': 'r2'
    }


def get_metrics(y_true, y_pred):
    """Calculate metrics for predictions"""
    mse_val = mean_squared_error(y_true, y_pred)
    return {
        "rmse": np.sqrt(mse_val),
        "r2": r2_score(y_true, y_pred)
    }


def base_models_dict():
    """Create dictionary of baseline models"""
    models = {
        "RF": RandomForestRegressor(**MODEL_PARAMS["RandomForest"]),
        "ENet": ElasticNet(**MODEL_PARAMS["ElasticNet"]),
        "Ridge": Ridge(**MODEL_PARAMS["Ridge"]),
        "Lasso": Lasso(**MODEL_PARAMS["Lasso"]),
        "SVR": SVR(**MODEL_PARAMS["SVR"])
    }

    if HAS_XGB:
        models["XGB"] = xgb.XGBRegressor(**MODEL_PARAMS["XGBoost"])
    if HAS_CAT:
        models["Cat"] = CatBoostRegressor(**MODEL_PARAMS["CatBoost"])
    if HAS_LGBM:
        models["LGBM"] = LGBMRegressor(**MODEL_PARAMS["LightGBM"])

    return models


def evaluate_single_model(name, model, X_train, y_train, X_test, y_test, feature_pipe, cv_splits=5):
    """Evaluate a single model with cross-validation"""
    try:
        # Create pipeline
        pipe = Pipeline([("features", clone(feature_pipe)), ("model", model)])

        # Cross-validation
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=get_scorers(), error_score='raise')

        cv_rmse = -scores["test_neg_rmse"].mean()
        cv_r2 = scores["test_r2"].mean()

        # Test set evaluation
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = get_metrics(y_test, y_pred)

        return {
            "model": name,
            "cv_rmse": cv_rmse,
            "cv_r2": cv_r2,
            "test_rmse": metrics["rmse"],
            "test_r2": metrics["r2"],
            "y_pred": y_pred,
            "status": "ok"
        }

    except Exception as e:
        print(f"❌ {name} error: {e}")
        traceback.print_exc()
        return {"model": name, "status": "fail", "error": str(e)}


def evaluate_all_models(models, X_train, y_train, X_test, y_test, feature_pipe, n_jobs=-1):
    """Evaluate all models in parallel"""
    print(f"🚀 Evaluating {len(models)} models...")

    # Parallel evaluation
    raw_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_single_model)(name, m, X_train, y_train, X_test, y_test, feature_pipe)
        for name, m in models.items()
    )

    # Filter successful results
    success_results = []
    for r in raw_results:
        if r.get("status") == "ok" and "y_pred" in r:
            success_results.append(r)

    if not success_results:
        print("⚠️ No models completed successfully.")
        return pd.DataFrame(), {}

    res_df = pd.DataFrame(success_results).drop(columns=["y_pred", "status"], errors='ignore').sort_values("cv_rmse")
    preds = {r["model"]: r["y_pred"] for r in success_results}

    return res_df, preds


def run_baseline_evaluation(models, X_train, y_train, X_test, y_test, feature_pipe, n_jobs=1):
    """Run baseline model evaluation"""
    print("=" * 50)
    print("BASELINE MODEL EVALUATION")
    print("=" * 50)

    res_df, preds = evaluate_all_models(models, X_train, y_train, X_test, y_test, feature_pipe, n_jobs)

    if res_df.empty:
        print("❌ ERROR: No models ran successfully!")
        return pd.DataFrame(), {}, []

    # Get top 5 models
    top5 = res_df["model"].head(5).tolist()
    best = res_df.iloc[0]

    print(f"\n🏆 Best Model: {best['model']}")
    print(f"   CV RMSE: {best['cv_rmse']:.5f}")
    print(f"   Test RMSE: {best['test_rmse']:.4f}")
    print(f"   Test R²: {best['test_r2']:.4f}")
    print(f"\n📊 Top 5 Models: {top5}")

    return res_df, preds, top5


def model_summary(results_df):
    """Print model evaluation summary"""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)
