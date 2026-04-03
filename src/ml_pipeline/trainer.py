"""
Model training module for the ML pipeline.
Handles model training, hyperparameter tuning with Optuna, and cross-validation.
"""

from typing import Dict, Optional, Tuple, List, Any
import numpy as np
import pandas as pd
from pathlib import Path
import traceback

import mlflow
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.svm import SVR
from joblib import Parallel, delayed

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

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from src.core.config import (
    RANDOM_STATE, CV_SPLITS, 
    RF_N_ESTIMATORS, RF_MAX_FEATURES,
    XGB_N_ESTIMATORS, XGB_LEARNING_RATE, XGB_MAX_DEPTH,
    CAT_N_ESTIMATORS, CAT_LEARNING_RATE, CAT_DEPTH,
    LGBM_N_ESTIMATORS, LGBM_LEARNING_RATE,
    ELASTICNET_ALPHA, ELASTICNET_L1_RATIO, RIDGE_ALPHA, LASSO_ALPHA,
    OPTUNA_N_TRIALS, LOGS_DIR
)
from src.core.utils import Logger, log_to_mlflow
from src.ml_pipeline.evaluation import get_scorers, get_metrics


logger = Logger(__name__)


def build_base_models() -> Dict[str, Any]:
    """
    Create a dictionary of baseline models.
    
    Returns:
        Dictionary mapping model names to model instances
    """
    models = {
        "RF": RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, 
            max_features=RF_MAX_FEATURES, 
            n_jobs=-1, 
            random_state=RANDOM_STATE
        ),
        "ENet": ElasticNet(
            alpha=ELASTICNET_ALPHA, 
            l1_ratio=ELASTICNET_L1_RATIO, 
            max_iter=20000, 
            random_state=RANDOM_STATE
        ),
        "Ridge": Ridge(
            alpha=RIDGE_ALPHA, 
            max_iter=20000, 
            random_state=RANDOM_STATE
        ),
        "Lasso": Lasso(
            alpha=LASSO_ALPHA, 
            max_iter=20000, 
            random_state=RANDOM_STATE
        ),
        "SVR": SVR(
            C=10.0, 
            epsilon=0.1
        )
    }

    if HAS_XGB:
        models["XGB"] = xgb.XGBRegressor(
            n_estimators=XGB_N_ESTIMATORS, 
            learning_rate=XGB_LEARNING_RATE, 
            max_depth=XGB_MAX_DEPTH, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            n_jobs=-1, 
            random_state=RANDOM_STATE
        )
    if HAS_CAT:
        models["Cat"] = CatBoostRegressor(
            n_estimators=CAT_N_ESTIMATORS, 
            learning_rate=CAT_LEARNING_RATE, 
            depth=CAT_DEPTH, 
            random_state=RANDOM_STATE, 
            verbose=False
        )
    if HAS_LGBM:
        models["LGBM"] = LGBMRegressor(
            n_estimators=LGBM_N_ESTIMATORS, 
            learning_rate=LGBM_LEARNING_RATE, 
            random_state=RANDOM_STATE, 
            n_jobs=-1, 
            verbosity=-1
        )

    logger.info(f"[OK] Initialized {len(models)} base models")
    return models


def evaluate_single_model(
    name: str,
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_pipe: Pipeline,
    cv_splits: int = CV_SPLITS
) -> Dict[str, Any]:
    """
    Evaluate a single model with cross-validation.
    
    Parameters:
        name: Model name
        model: Model instance
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_pipe: Feature preprocessing pipeline
        cv_splits: Number of CV folds
        
    Returns:
        Dictionary with evaluation results
    """
    try:
        with mlflow.start_run(run_name=f"Eval_{name}", nested=True):
            # Create pipeline
            pipe = Pipeline([("features", clone(feature_pipe)), ("model", model)])

            # Cross-validation
            cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_validate(pipe, X_train, y_train, cv=cv, 
                                   scoring=get_scorers(), error_score='raise')

            cv_rmse = -scores["test_neg_rmse"].mean()
            cv_r2 = scores["test_r2"].mean()

            # Test set evaluation
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            metrics = get_metrics(y_test, y_pred)

            # Log to MLflow
            mlflow.log_params({"model": name, "cv_splits": cv_splits})
            mlflow.log_metrics({
                "cv_rmse": cv_rmse, "cv_r2": cv_r2,
                "test_rmse": metrics["rmse"], "test_r2": metrics["r2"]
            })

            logger.info(f"[OK] {name}: RMSE={cv_rmse:.4f}, R²={cv_r2:.4f}")
            
            return {
                "model": name, "cv_rmse": cv_rmse, "cv_r2": cv_r2,
                "test_rmse": metrics["rmse"], "test_r2": metrics["r2"],
                "y_pred": y_pred, "status": "ok"
            }

    except Exception as e:
        logger.error(f"✗ {name} failed: {e}")
        traceback.print_exc()
        return {"model": name, "status": "fail", "error": str(e)}


def evaluate_all_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_pipe: Pipeline,
    n_jobs: int = -1
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Evaluate all models in parallel.
    
    Parameters:
        models: Dictionary of models to evaluate
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_pipe: Feature preprocessing pipeline
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (results_dataframe, predictions_dict)
    """
    logger.info(f"[START] Evaluating {len(models)} models...")

    raw_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_single_model)(name, m, X_train, y_train, X_test, y_test, feature_pipe)
        for name, m in models.items()
    )

    # Filter successful results
    success_results = [r for r in raw_results if r.get("status") == "ok" and "y_pred" in r]

    if not success_results:
        logger.error("[FAIL] No models completed successfully")
        return pd.DataFrame(), {}

    res_df = pd.DataFrame(success_results).drop(columns=["y_pred", "status"], errors='ignore').sort_values("cv_rmse")
    preds = {r["model"]: r["y_pred"] for r in success_results}

    logger.info(f"[OK] Evaluation complete: {len(success_results)} successful")
    return res_df, preds


def run_baseline_evaluation(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_pipe: Pipeline,
    n_jobs: int = 1
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], List[str]]:
    """
    Run baseline model evaluation.
    
    Parameters:
        models: Dictionary of models
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        feature_pipe: Feature preprocessing pipeline
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (results_df, predictions, top5_models)
    """
    with mlflow.start_run(run_name="Baseline_Evaluation", nested=True):
        res_df, preds = evaluate_all_models(models, X_train, y_train, X_test, y_test, 
                                           feature_pipe, n_jobs)

        if res_df.empty:
            logger.error("[FAIL] No models evaluated successfully")
            return pd.DataFrame(), {}, []

        top5 = res_df["model"].head(5).tolist()
        best = res_df.iloc[0]

        # Log to MLflow
        mlflow.log_params({"best_model": best['model'], "top5": str(top5)})
        mlflow.log_metrics({
            "best_cv_rmse": best['cv_rmse'],
            "best_test_rmse": best['test_rmse']
        })

        # Save results
        results_path = LOGS_DIR / "baseline_results.csv"
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not create logs directory {LOGS_DIR}: {e}")
        
        try:
            res_df.to_csv(results_path, index=False)
            mlflow.log_artifact(str(results_path))
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not save results to {results_path}: {e}")

        logger.info(f"[RANKING] Best model: {best['model']} | RMSE: {best['cv_rmse']:.4f}")
        return res_df, preds, top5


def make_objective_with_mlflow(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_pipe: Pipeline
):
    """
    Create Optuna objective function for hyperparameter tuning.
    
    Parameters:
        name: Model name
        X_train: Training features
        y_train: Training target
        feature_pipe: Feature preprocessing pipeline
        
    Returns:
        Objective function for Optuna
    """
    def objective(trial):
        with mlflow.start_run(run_name=f"Trial_{name}_{trial.number}", nested=True):
            # Define search space
            if name == "RF":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 600, 3000, step=200),
                    "max_depth": trial.suggest_int("max_depth", 3, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                    "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1
                }
                model = RandomForestRegressor(**params)

            elif name == "Cat":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 400, 3000, step=500),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                    "random_state": RANDOM_STATE, "verbose": False, "loss_function": 'RMSE'
                }
                model = CatBoostRegressor(**params)

            elif name == "XGB":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 400, 3000, step=500),
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

            elif name == "LGBM":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 400, 3000, step=500),
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
                params = {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True), 
                         "random_state": RANDOM_STATE}
                if name == "Ridge":
                    model = Ridge(**params)
                elif name == "Lasso":
                    model = Lasso(**params)
                else:
                    params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.1, 0.9)
                    model = ElasticNet(**params)
            else:
                raise ValueError(f"Model {name} not configured")

            # Build pipeline
            pipe = Pipeline([("features", clone(feature_pipe)), ("model", model)])

            # Cross-validation
            cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_validate(pipe, X_train, y_train, cv=cv, 
                                   scoring=get_scorers(), n_jobs=2)

            cv_rmse = -scores["test_neg_rmse"].mean()
            cv_r2 = scores["test_r2"].mean()

            # Log to MLflow
            mlflow.log_params(trial.params)
            mlflow.log_metrics({"cv_rmse": cv_rmse, "cv_r2": cv_r2})

            return cv_rmse

    return objective


def tune_single_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_pipe: Pipeline,
    n_trials: int = OPTUNA_N_TRIALS
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """
    Tune a single model using Optuna.
    
    Parameters:
        name: Model name
        X_train: Training features
        y_train: Training target
        feature_pipe: Feature preprocessing pipeline
        n_trials: Number of trials
        
    Returns:
        Tuple of (tuned_model_info, trials_dataframe)
    """
    if not HAS_OPTUNA:
        logger.warning("Optuna not installed, skipping tuning")
        return None, None

    logger.info(f"[TUNE] Tuning {name} ({n_trials} trials)...")

    with mlflow.start_run(run_name=f"Tuning_{name}", nested=True):
        study = optuna.create_study(direction="minimize")
        objective = make_objective_with_mlflow(name, X_train, y_train, feature_pipe)
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        # Get base model and update with best params
        base_models = build_base_models()
        if name not in base_models:
            logger.warning(f"Model {name} not in base_models")
            return None, None
            
        best_model = base_models[name]
        clean_params = {k: v for k, v in study.best_params.items() 
                       if k not in ['n_jobs', 'random_state']}
        best_model.set_params(**clean_params)

        # Log results
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_rmse", study.best_value)

        logger.info(f"[OK] {name} tuning complete: RMSE={study.best_value:.4f}")
        
        return {
            "estimator": best_model, 
            "best_cv_rmse": study.best_value, 
            "best_params": study.best_params
        }, study.trials_dataframe()


def run_hyperparameter_tuning(
    top_names: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_pipe: Pipeline,
    n_trials: int = OPTUNA_N_TRIALS
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, pd.DataFrame]]:
    """
    Orchestrate hyperparameter tuning for multiple models.
    
    Parameters:
        top_names: List of model names to tune
        X_train: Training features
        y_train: Training target
        feature_pipe: Feature preprocessing pipeline
        n_trials: Number of trials per model
        
    Returns:
        Tuple of (tuned_models_dict, trials_history_dict)
    """
    tuned, histories = {}, {}
    logger.info(f"[START] Starting tuning for: {top_names}")

    with mlflow.start_run(run_name="Tuning_Summary", nested=True):
        for name in top_names:
            result, history = tune_single_model(name, X_train, y_train, feature_pipe, n_trials)
            if result:
                tuned[name], histories[name] = result, history

    logger.info(f"[OK] Tuning complete: {len(tuned)} models tuned")
    return tuned, histories
