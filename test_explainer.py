"""
Quick test to verify SHAP explainer was created and saved correctly.
Run this after training pipeline to check if explainer artifacts exist.
"""

from pathlib import Path
import joblib
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.core.config import EXPLAINER_DIR, MODELS_DIR


def check_explainer_artifacts():
    """Check if SHAP explainer artifacts exist and are valid."""
    
    logger.info("=" * 80)
    logger.info("CHECKING EXPLAINER ARTIFACTS")
    logger.info("=" * 80)
    
    # Check directory exists
    if not EXPLAINER_DIR.exists():
        logger.error(f"[FAIL] Explainer directory does not exist: {EXPLAINER_DIR}")
        assert False, "Explainer directory does not exist"

    
    logger.info(f"[OK] Explainer directory exists: {EXPLAINER_DIR}")
    
    # List files in directory
    files = list(EXPLAINER_DIR.glob("*"))
    logger.info(f"\nFiles in {EXPLAINER_DIR}:")
    if not files:
        logger.error("[FAIL] Directory is empty!")
        assert False, "Directory is empty!"
    
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  - {f.name} ({size_mb:.2f} MB)")
    
    # Try to load main explainer file
    explainer_file = EXPLAINER_DIR / "shap_explainer.joblib"
    if not explainer_file.exists():
        logger.error(f"[FAIL] Main explainer file not found: {explainer_file}")
        assert False, "Main explainer file not found"
    
    logger.info(f"\n[OK] Main explainer file found: {explainer_file}")
    
    try:
        logger.info("[DEBUG] Loading explainer data...")
        explainer_data = joblib.load(explainer_file)
        
        logger.info(f"[OK] Explainer loaded successfully")
        logger.info(f"\nExplainer data structure:")
        
        for key, value in explainer_data.items():
            if hasattr(value, 'shape'):
                logger.info(f"  - {key}: {type(value).__name__} shape={value.shape}")
            elif isinstance(value, list):
                logger.info(f"  - {key}: {type(value).__name__} length={len(value)}")
            else:
                logger.info(f"  - {key}: {type(value).__name__}")
        
        # Verify required fields
        required_fields = ['explainer', 'X_background', 'feature_names_transformed']
        missing = [f for f in required_fields if f not in explainer_data]
        
        if missing:
            logger.error(f"[FAIL] Missing required fields: {missing}")
            assert False, "Missing required fields"
        
        logger.info(f"\n[OK] All required fields present")
        
        # Check feature names
        feature_names = explainer_data.get('feature_names_transformed', [])
        logger.info(f"\nFeature count: {len(feature_names)}")
        logger.info(f"First 10 features: {feature_names[:10]}")
        
        # Check model
        model_obj = explainer_data.get('explainer')
        if model_obj is not None:
            logger.info(f"[OK] SHAP TreeExplainer loaded, type: {type(model_obj).__name__}")
        
        logger.info("\n" + "=" * 80)
        logger.info("[SUCCESS] All explainer artifacts verified successfully!")
        logger.info("=" * 80)
        assert True, "Explainer artifacts verified successfully"
        
    except Exception as e:
        logger.error(f"[FAIL] Failed to load explainer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        assert False, "Failed to load explainer"


def check_models():
    """Check if models exist."""
    logger.info("\n\nCHECKING MODELS")
    logger.info("-" * 80)
    
    if not MODELS_DIR.exists():
        logger.warning(f"Models directory not found: {MODELS_DIR}")
        assert False, "Models directory not found"
    
    model_files = list(MODELS_DIR.glob("*.joblib")) + list(MODELS_DIR.glob("*.pkl"))
    if not model_files:
        logger.warning("No model files found")
        assert False, "No model files found"
    
    for f in model_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  - {f.name} ({size_mb:.2f} MB)")
    
    assert True, "Models verified successfully"


if __name__ == "__main__":
    check_models()
    success = check_explainer_artifacts()
    exit(0 if success else 1)
