#!/usr/bin/env python3
# ============================================================================
# Hyperparameter Tuning with Optuna
# ============================================================================
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from joblib import Parallel, delayed
import traceback

from modules.config import RANDOM_STATE, LOGS_DIR, USE_MLFLOW
from modules.models import base_models_dict, get_scorers
from modules.cache_manager import MLflowTrainingCacheManager

# MLflow integration
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# Optuna integration
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    print("❌ ERROR: Optuna not installed!")
    print("   Install with: pip install optuna")
    sys.exit(1)

# Optional boosting libraries
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

from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


def make_objective_with_mlflow(name, X_train, y_train, feature_pipe):
    """Create Optuna objective function for model tuning"""
    
    def objective(trial):
        try:
            with mlflow.start_run(run_name=f"Trial_{name}_{trial.number}", nested=True) if HAS_MLFLOW else nullcontext():
                
                # Define hyperparameter search space
                if name == "RF":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=200),
                        "max_depth": trial.suggest_int("max_depth", 3, 20),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                        "random_state": RANDOM_STATE,
                        "n_jobs": -1
                    }
                    model = RandomForestRegressor(**params)

                elif name == "Cat" and HAS_CAT:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 400, 2000, step=500),
                        "depth": trial.suggest_int("depth", 4, 10),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                        "random_state": RANDOM_STATE,
                        "verbose": False,
                        "loss_function": 'RMSE'
                    }
                    model = CatBoostRegressor(**params)

                elif name == "XGB" and HAS_XGB:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 400, 2000, step=500),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                        "max_depth": trial.suggest_int("max_depth", 3, 7),
                        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
                        "tree_method": "hist",
                        "random_state": RANDOM_STATE,
                        "n_jobs": 1
                    }
                    model = xgb.XGBRegressor(**params)

                elif name == "LGBM" and HAS_LGBM:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 400, 2000, step=500),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                        "max_depth": trial.suggest_int("max_depth", 3, 12),
                        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                        "random_state": RANDOM_STATE,
                        "n_jobs": 1,
                        "verbosity": -1
                    }
                    model = LGBMRegressor(**params)

                elif name in ["Ridge", "Lasso", "ENet"]:
                    params = {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True), "random_state": RANDOM_STATE}
                    if name == "Ridge":
                        model = Ridge(**params)
                    elif name == "Lasso":
                        model = Lasso(max_iter=20000, **params)
                    else:
                        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.1, 0.9)
                        model = ElasticNet(max_iter=20000, **params)

                else:
                    raise ValueError(f"Model {name} not supported.")

                # Build pipeline
                pipe = Pipeline([
                    ("features", clone(feature_pipe)),
                    ("model", model)
                ])

                # Cross-validation
                cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                scores = cross_validate(
                    pipe, X_train, y_train,
                    cv=cv,
                    scoring=get_scorers(),
                    n_jobs=2
                )

                # Calculate metrics
                cv_rmse = -scores["test_neg_rmse"].mean()
                cv_r2 = scores["test_r2"].mean()

                # Log to MLflow
                if HAS_MLFLOW:
                    try:
                        mlflow.log_params(trial.params)
                        mlflow.log_metrics({"cv_rmse": cv_rmse, "cv_r2": cv_r2})
                    except:
                        pass

                return cv_rmse  # Optuna minimizes this

        except Exception as e:
            print(f"  Trial failed: {e}")
            return float('inf')

    return objective


def tune_single_model(name, X_train, y_train, feature_pipe, n_trials=20):
    """Tune a single model with Optuna"""
    
    if not HAS_OPTUNA:
        return None, None

    print(f"\n🎯 Tuning {name} ({n_trials} trials)...")

    study = optuna.create_study(direction="minimize")
    objective = make_objective_with_mlflow(name, X_train, y_train, feature_pipe)
    
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    except KeyboardInterrupt:
        print(f"  ⏸️  Tuning interrupted by user")
        return study, study.trials_dataframe()

    # Get best parameters
    best_params = study.best_params
    print(f"✅ Best params for {name}:")
    print(f"   CV RMSE: {study.best_value:.4f}")
    for k, v in list(best_params.items())[:3]:
        print(f"   {k}: {v}")

    return study, study.trials_dataframe()


def run_hyperparameter_tuning(top_names, X_train, y_train, feature_pipe, n_trials=15):
    """Run hyperparameter tuning for top models"""
    
    print("\n" + "=" * 70)
    print("🔧 HYPERPARAMETER TUNING WITH OPTUNA")
    print("=" * 70)
    print(f"Models to tune: {top_names}")
    print(f"Trials per model: {n_trials}")

    tuned = {}
    histories = {}

    if USE_MLFLOW and HAS_MLFLOW:
        with mlflow.start_run(run_name="Tuning_Summary"):
            for name in top_names:
                study, history = tune_single_model(name, X_train, y_train, feature_pipe, n_trials)
                if study:
                    tuned[name] = study.best_params
                    histories[name] = history
    else:
        for name in top_names:
            study, history = tune_single_model(name, X_train, y_train, feature_pipe, n_trials)
            if study:
                tuned[name] = study.best_params
                histories[name] = history

    print(f"\n✅ Tuning complete! Tuned models: {list(tuned.keys())}")
    return tuned, histories


# Context manager for MLflow
from contextlib import nullcontext


def main(X_train, y_train, feature_pipe, top_names=None):
    """Main tuning pipeline"""
    
    if top_names is None:
        top_names = ["RF", "Ridge", "Lasso"]
        if HAS_XGB:
            top_names.append("XGB")
        if HAS_CAT:
            top_names.append("Cat")
        if HAS_LGBM:
            top_names.append("LGBM")

    tuned, histories = run_hyperparameter_tuning(
        top_names, X_train, y_train, feature_pipe, n_trials=15
    )

    # Save results
    cache = MLflowTrainingCacheManager(LOGS_DIR / "tuning_cache")
    cache.save_results({'tuned': tuned, 'histories': histories}, 'tuning_results_local')

    return tuned, histories


if __name__ == "__main__":
    print("Use this module within train.py for hyperparameter tuning.")
    print("It requires pre-trained baseline results.")
