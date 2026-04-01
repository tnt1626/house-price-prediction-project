"""
Explainability module for model interpretation using SHAP.
Handles SHAP explainer creation, training, and local explanations.
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import logging
import re
from sklearn.pipeline import Pipeline

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from src.core.config import EXPLAINER_DIR
from src.core.utils import Logger


logger = Logger(__name__)


class ModelExplainer:
    """
    SHAP-based model explainer for local and global explanations.
    Supports tree-based models (CatBoost, XGBoost, LightGBM, Random Forest).
    """
    
    def __init__(self, model: Any, preprocessor: Any, background_size: int = 100, original_feature_names: List[str] = None):
        """
        Initialize ModelExplainer.
        
        Parameters:
            model: Trained tree-based model (CatBoost, XGBoost, etc.)
            preprocessor: Fitted ColumnTransformer or preprocessing pipeline
            background_size: Number of samples for SHAP background data
            original_feature_names: List of original feature names from schema
        """
        if not HAS_SHAP:
            raise ImportError("SHAP is required for explainability. Install with: pip install shap")
        
        self.model = model
        self.preprocessor = preprocessor
        self.background_size = background_size
        self.explainer = None
        self.X_background = None
        self.feature_names_original = original_feature_names or []
        self.feature_names_transformed = None
        
        logger.info("[OK] ModelExplainer initialized")
    
    def _get_feature_names_original(self, X: pd.DataFrame) -> List[str]:
        """
        Extract original feature names from preprocessor.
        Maps transformed feature names back to original input names.
        """
        try:
            # Get feature names after preprocessing
            transformed_names = self._get_feature_names_transformed()
            
            # Get original input column names
            original_names = X.columns.tolist()
            
            # Map back using feature spec if available
            # This is preprocessor-dependent
            if hasattr(self.preprocessor, 'transformers_'):
                feature_mapping = {}
                
                for name, transformer, columns in self.preprocessor.transformers_:
                    if name == 'remainder':
                        continue
                    
                    if hasattr(transformer, 'get_feature_names_out'):
                        try:
                            output_names = transformer.get_feature_names_out(columns)
                            for orig, trans in zip(columns, output_names):
                                feature_mapping[trans] = orig
                        except Exception as e:
                            logger.warning(f"Failed to get feature names from {name}: {e}")
                            for col in columns:
                                feature_mapping[col] = col
                    else:
                        for col in columns:
                            feature_mapping[col] = col
                
                return feature_mapping
            
            return {name: name for name in original_names}
            
        except Exception as e:
            logger.warning(f"Failed to map feature names: {e}")
            return {name: name for name in X.columns}
    
    def _get_feature_names_transformed(self) -> List[str]:
        """
        Trích xuất tên đặc trưng sau khi biến đổi một cách mạnh mẽ.
        Hỗ trợ Pipeline lồng nhau và ColumnTransformer.
        """
        try:
            logger.info(f"[DEBUG] Attempting to extract feature names from: {type(self.preprocessor).__name__}")
            
            # 1. Thử gọi trực tiếp (Cách chuẩn của Scikit-Learn hiện đại)
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                try:
                    names = self.preprocessor.get_feature_names_out().tolist()
                    if names and len(names) > 0:
                        logger.info(f"[OK] Got {len(names)} names from get_feature_names_out()")
                        return names
                except Exception as e:
                    logger.debug(f"Direct get_feature_names_out failed: {e}")

            # 2. Xử lý Pipeline lồng nhau (drill down)
            # Theo cấu trúc của bạn: Pipeline -> Step('preproc') -> Pipeline -> Step('ct')
            current_step = self.preprocessor
            
            # Đi sâu vào bước 'preproc' nếu có
            if hasattr(current_step, 'named_steps') and 'preproc' in current_step.named_steps:
                current_step = current_step.named_steps['preproc']
                
            # Đi sâu vào bước 'ct' (ColumnTransformer) nếu có
            if hasattr(current_step, 'named_steps') and 'ct' in current_step.named_steps:
                current_step = current_step.named_steps['ct']

            # 3. Trích xuất thủ công từ ColumnTransformer nếu các bước trên thất bại
            if hasattr(current_step, 'transformers_'):
                logger.info("[DEBUG] Extracting features manually from ColumnTransformer")
                names = []
                for trans_name, transformer, columns in current_step.transformers_:
                    if trans_name == 'remainder' and transformer == 'drop':
                        continue
                    
                    if hasattr(transformer, 'get_feature_names_out'):
                        try:
                            # Lấy tên cột đã biến đổi (ví dụ: OneHot thành Neighborhood_StoneBr)
                            out_names = transformer.get_feature_names_out(columns)
                            names.extend(out_names)
                        except Exception:
                            names.extend([f"{trans_name}__{col}" for col in columns])
                    else:
                        # Nếu là các bước passthrough hoặc mapper đơn giản
                        names.extend(columns)
                
                if names:
                    return names

            logger.warning("[WARN] Could not determine feature names, returning empty list")
            return []
            
        except Exception as e:
            logger.warning(f"[WARN] Exception while extracting feature names: {e}")
            return []
    
    def _clean_feature_name(self, feature_name: str) -> str:
        """
        Clean feature name by removing transformer prefixes.
        Examples:
            - 'TargetEncoder_Neighborhood' -> 'Neighborhood'
            - 'OneHotEncoder_BldgType_1Fam' -> 'BldgType'
            - 'x0', 'x1' -> generic names
        """
        # Remove common sklearn prefixes
        prefixes = [
            'TargetEncoder_', 'OneHotEncoder_', 'OrdinalEncoder_',
            'StandardScaler_', 'MinMaxScaler_', 'RobustScaler_',
            'QuantileTransformer_', 'PowerTransformer_',
            'remainder__', 'passthrough_', 'cat__', 'num_cont__', 
            'num_abs__', 'ord__',
        ]
        
        cleaned = feature_name
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned.replace(prefix, '', 1)
                break

        return cleaned
    
    def fit(self, X_train: pd.DataFrame, y_train: Optional[np.ndarray] = None) -> None:
        """
        Fit SHAP explainer on training data.
        
        Parameters:
            X_train: Preprocessed training features (already transformed)
            y_train: Optional target values (for future use)
        """
        try:
            # Convert to numpy if necessary
            if isinstance(X_train, pd.DataFrame):
                X_train_np = X_train.values
                self.feature_names_transformed = X_train.columns.tolist()
                logger.info(f"[DEBUG] Feature names from DataFrame: {self.feature_names_transformed[:5]}...")
            else:
                X_train_np = X_train
                if self.feature_names_transformed is None:
                    self.feature_names_transformed = self._get_feature_names_transformed()
            
            # If feature names still empty, generate generic names
            if not self.feature_names_transformed or len(self.feature_names_transformed) == 0:
                n_features = X_train_np.shape[1]
                self.feature_names_transformed = [f"feature_{i}" for i in range(n_features)]
                logger.warning(f"[WARN] Generated generic feature names: {n_features} features")
            
            # Ensure feature names match data shape
            if len(self.feature_names_transformed) != X_train_np.shape[1]:
                logger.warning(
                    f"[WARN] Feature name count ({len(self.feature_names_transformed)}) "
                    f"doesn't match data shape ({X_train_np.shape[1]})"
                )
                self.feature_names_transformed = [f"feature_{i}" for i in range(X_train_np.shape[1])]
            
            # Select background samples for efficiency
            if X_train_np.shape[0] > self.background_size:
                indices = np.random.choice(
                    X_train_np.shape[0],
                    size=self.background_size,
                    replace=False
                )
                self.X_background = X_train_np[indices]
                logger.info(f"[DEBUG] Selected {self.background_size} background samples from {X_train_np.shape[0]}")
            else:
                self.X_background = X_train_np
                logger.info(f"[DEBUG] Using all {X_train_np.shape[0]} samples as background")
            
            # Create SHAP explainer
            # Using TreeExplainer for tree-based models (most efficient)
            logger.info(f"[DEBUG] Creating TreeExplainer with model type: {type(self.model).__name__}")
            self.explainer = shap.TreeExplainer(self.model)
            
            logger.info(
                f"[OK] SHAP explainer fitted successfully with background size: {self.X_background.shape[0]}"
            )
            logger.info(f"[OK] Feature names: {self.feature_names_transformed[:10]}")
            
        except Exception as e:
            logger.error(f"Failed to fit explainer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def get_local_explanation(
        self,
        X_single: pd.DataFrame,
        original_input: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Get local SHAP explanation for a single sample.
        
        Parameters:
            X_single: Single preprocessed sample (shape: 1, n_features) - DataFrame or numpy array
            original_input: Original input dict for feature name mapping
            top_k: Number of top features to return
            
        Returns:
            Dictionary with base_value, predicted_value, and feature explanations
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not fitted. Call fit() first.")
        
        try:
            # IMPORTANT: Keep track of original format and ensure DataFrame is available for column access
            X_df_original = None
            if isinstance(X_single, pd.DataFrame):
                X_df_original = X_single
                X_np = X_single.values
            else:
                X_np = X_single
                # Try to create DataFrame from numpy array if we have feature names
                if self.feature_names_transformed and len(self.feature_names_transformed) > 0:
                    try:
                        if isinstance(X_np, np.ndarray) and X_np.ndim == 1:
                            X_np = X_np.reshape(1, -1)
                        X_df_original = pd.DataFrame(X_np, columns=self.feature_names_transformed)
                    except Exception as e:
                        logger.warning(f"Could not create DataFrame from numpy array: {e}")
            
            if X_np.ndim == 1:
                X_np = X_np.reshape(1, -1)
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X_np)
            
            # Handle multi-output models
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Use first output
            
            shap_values = shap_values[0]  # Single sample
            
            # Get base value (expected value)
            base_value = float(self.explainer.expected_value)
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
            
            # Get model prediction
            try:
                # If model is a Pipeline with preprocessing, extract just the final estimator
                # because X_np is already transformed data from SHAP
                prediction_model = self.model
                if isinstance(self.model, Pipeline):
                    prediction_model = self.model.named_steps.get('model') or self.model.named_steps[list(self.model.named_steps.keys())[-1]]
                
                # Use numpy array (already transformed) for the final model
                prediction = float(prediction_model.predict(X_np)[0])
            except Exception as e:
                logger.warning(f"Predict failed: {e}")
                # Fallback: try with the original model in case it handles both
                try:
                    prediction = float(self.model.predict(X_np)[0])
                except Exception as e2:
                    logger.error(f"Final prediction failed: {e2}")
                    raise
            
            # Prepare feature explanations
            explanations = []
            
            for idx, (shap_val, feature_name) in enumerate(zip(shap_values, self.feature_names_transformed or [])):
                # Clean feature name
                clean_name = self._clean_feature_name(feature_name)
                
                # Try to map to original feature name if available
                display_name = clean_name
                if idx < len(self.feature_names_original) and self.feature_names_original[idx]:
                    display_name = self.feature_names_original[idx]
                
                # Get original value if available
                original_value = None
                
                # Try to get from original_input dict first (try both display name and clean name)
                if original_input:
                    if display_name in original_input:
                        original_value = original_input[display_name]
                    elif clean_name in original_input:
                        original_value = original_input[clean_name]
                    elif feature_name in original_input:
                        original_value = original_input[feature_name]
                
                # If not found, try to get from DataFrame
                if original_value is None and X_df_original is not None:
                    try:
                        if feature_name in X_df_original.columns:
                            original_value = float(X_df_original[feature_name].values[0])
                        elif clean_name in X_df_original.columns:
                            original_value = float(X_df_original[clean_name].values[0])
                    except Exception as e:
                        logger.debug(f"Could not extract original value for {feature_name}: {e}")
                
                # Determine contribution type
                contribution_type = "positive" if shap_val > 0 else "negative"
                
                explanations.append({
                    "feature_name": display_name,
                    "original_value": original_value,
                    "shap_value": float(shap_val),
                    "contribution_type": contribution_type,
                    "abs_shap_value": float(abs(shap_val))
                })
            
            # Sort by absolute SHAP value and take top_k
            explanations.sort(key=lambda x: x['abs_shap_value'], reverse=True)
            explanations = explanations[:top_k]
            
            # Remove abs_shap_value from output
            for exp in explanations:
                del exp['abs_shap_value']
            
            return {
                "base_value": base_value,
                "predicted_value": prediction,
                "explanations": explanations
            }
            
        except Exception as e:
            logger.error(f"Failed to get local explanation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save explainer to disk.
        
        Parameters:
            path: Path to save explainer (default: EXPLAINER_DIR/explainer.joblib)
            
        Returns:
            Path to saved file
        """
        if self.explainer is None:
            raise RuntimeError("Explainer not fitted. Cannot save.")
        
        if path is None:
            EXPLAINER_DIR.mkdir(parents=True, exist_ok=True)
            path = EXPLAINER_DIR / "shap_explainer.joblib"
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"[DEBUG] Saving explainer to: {path}")
            
            # Create a serializable object
            explainer_data = {
                'explainer': self.explainer,
                'X_background': self.X_background,
                'feature_names_transformed': self.feature_names_transformed,
                'feature_names_original': self.feature_names_original,
                'background_size': self.background_size
            }
            
            logger.info(f"[DEBUG] Explainer data keys: {explainer_data.keys()}")
            logger.info(f"[DEBUG] X_background shape: {self.X_background.shape if self.X_background is not None else None}")
            logger.info(f"[DEBUG] Feature names count: {len(self.feature_names_transformed) if self.feature_names_transformed else 0}")
            logger.info(f"[DEBUG] Original feature names count: {len(self.feature_names_original) if self.feature_names_original else 0}")
            
            # Save with joblib
            joblib.dump(explainer_data, path, compress=3)
            
            # Verify file was created
            if not path.exists():
                raise RuntimeError(f"File was not created at {path}")
            
            file_size = path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"[OK] Explainer saved successfully to {path} ({file_size:.2f} MB)")
            return path
            
        except Exception as e:
            logger.error(f"Failed to save explainer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    @classmethod
    def load(cls, model: Any, preprocessing: Any, path: Optional[Path] = None, original_feature_names: List[str] = None) -> 'ModelExplainer':
        """
        Load explainer from disk.
        
        Parameters:
            model: Trained model (required for initialization)
            preprocessing: Fitted preprocessor (required for initialization)
            path: Path to explainer file (default: EXPLAINER_DIR/shap_explainer.joblib)
            original_feature_names: List of original feature names from schema (optional override)
            
        Returns:
            Loaded ModelExplainer instance
        """
        if path is None:
            path = EXPLAINER_DIR / "shap_explainer.joblib"
        else:
            path = Path(path)
        
        if not path.exists():
            logger.error(f"Explainer file not found: {path}")
            raise FileNotFoundError(f"Explainer file not found: {path}")
        
        try:
            logger.info(f"[DEBUG] Loading explainer from: {path}")
            explainer_data = joblib.load(path)
            
            logger.info(f"[DEBUG] Loaded explainer data keys: {explainer_data.keys()}")
            
            # Use provided original_feature_names if available, otherwise use saved ones
            names_to_use = original_feature_names or explainer_data.get('feature_names_original')
            
            # Create instance
            instance = cls(model, preprocessing, original_feature_names=names_to_use)
            instance.explainer = explainer_data['explainer']
            instance.X_background = explainer_data['X_background']
            instance.feature_names_transformed = explainer_data['feature_names_transformed']
            
            logger.info(f"[OK] Explainer loaded successfully from {path}")
            logger.info(f"[DEBUG] X_background shape: {instance.X_background.shape}")
            logger.info(f"[DEBUG] Feature names count: {len(instance.feature_names_transformed) if instance.feature_names_transformed else 0}")
            logger.info(f"[DEBUG] Original feature names count: {len(instance.feature_names_original) if instance.feature_names_original else 0}")
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to load explainer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
