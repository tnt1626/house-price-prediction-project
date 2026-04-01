"""
Test script to verify column name mapping works correctly.
Checks that schema names are properly mapped to data column names.
"""

from pathlib import Path
import pandas as pd
import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.core.config import SCHEMA_TO_DATA_MAPPING, TRAIN_DATA_FILE
from src.api.schemas import HousePriceInput
from src.api.services import PredictionService


def test_column_mapping():
    """Test that input data maps correctly to actual column names."""
    
    logger.info("=" * 80)
    logger.info("TESTING COLUMN NAME MAPPING")
    logger.info("=" * 80)
    
    # Load actual data to see column names
    if TRAIN_DATA_FILE.exists():
        logger.info(f"\n[DEBUG] Loading actual data from {TRAIN_DATA_FILE}")
        df_actual = pd.read_csv(TRAIN_DATA_FILE, nrows=1)
        logger.info(f"[OK] Actual data columns: {df_actual.columns.tolist()}")
    else:
        logger.warning(f"[WARN] Data file not found: {TRAIN_DATA_FILE}")
        df_actual = None
    
    # Create test input using schema
    logger.info("\n[DEBUG] Creating test input from schema")
    test_input = HousePriceInput(
        LotArea=8450,
        OverallQual=7,
        OverallCond=5,
        YearBuilt=2003,
        YearRemodAdd=2003,
        TotalBsmtSF=1000,
        FirstFlrSF=856,  # Schema name
        SecondFlrSF=854,  # Schema name
        GrLivArea=1710,
        FullBath=2,
        HalfBath=1,
        BsmtFullBath=1,
        BsmtHalfBath=0,
        Bedroom=3,  # Schema name
        Kitchen=1,  # Schema name
        TotRmsAbvGrd=8,
        Fireplaces=1,
        GarageCars=2,
        GarageSF=548,
        MSZoning="RL",
        Neighborhood="NAmes",
        BldgType="1Fam",
        ExterQual="Gd",
        ExterCond="TA",
        BsmtQual="Gd",
        BsmtCond="TA",
        HeatingQC="Ex",
        KitchenQual="Gd",
        FireplaceQu="Ta",
        GarageQual="TA",
        GarageCond="TA",
        MoSold=7,
        YrSold=2022,
        SaleType="WD",
        SaleCondition="Normal"
    )
    
    logger.info(f"[OK] Test input created")
    logger.info(f"[DEBUG] Schema input fields: FirstFlrSF, SecondFlrSF, Bedroom, Kitchen")
    
    # Convert to DataFrame using PredictionService method
    service = PredictionService()
    df_from_input = service.input_to_dataframe(test_input)
    
    logger.info(f"\n[DEBUG] DataFrame from schema input:")
    logger.info(f"  Columns: {df_from_input.columns.tolist()}")
    
    # Check mappings applied
    mapped_cols = [v for k, v in SCHEMA_TO_DATA_MAPPING.items() if k in df_from_input.columns]
    if mapped_cols:
        logger.info(f"[OK] Mapped columns: {mapped_cols}")
    else:
        logger.warning(f"[WARN] No columns were mapped")
    
    # Verify specific mappings
    expected_mappings = {
        "1stFlrSF": test_input.FirstFlrSF,
        "2ndFlrSF": test_input.SecondFlrSF,
        "BedroomAbvGr": test_input.Bedroom,
        "KitchenAbvGr": test_input.Kitchen,
    }
    
    logger.info(f"\n[DEBUG] Verifying value mappings:")
    all_good = True
    for data_col, expected_val in expected_mappings.items():
        if data_col in df_from_input.columns:
            actual_val = df_from_input[data_col].values[0]
            if actual_val == expected_val:
                logger.info(f"  ✓ {data_col}: {actual_val} == {expected_val}")
            else:
                logger.error(f"  ✗ {data_col}: {actual_val} != {expected_val}")
                all_good = False
        else:
            logger.error(f"  ✗ {data_col}: NOT FOUND in DataFrame")
            all_good = False
    
    # Compare with actual data if available
    if df_actual is not None:
        logger.info(f"\n[DEBUG] Comparing with actual data column names:")
        for schema_name, data_name in SCHEMA_TO_DATA_MAPPING.items():
            in_actual = data_name in df_actual.columns
            in_mapped = data_name in df_from_input.columns
            logger.info(f"  {schema_name} -> {data_name}: actual={in_actual}, mapped={in_mapped}")
    
    logger.info("\n" + "=" * 80)
    if all_good:
        logger.info("[SUCCESS] All column mappings verified!")
    else:
        logger.error("[FAILED] Some column mappings failed!")
    logger.info("=" * 80)
    
    assert all_good, "Column mapping verification failed!"


if __name__ == "__main__":
    success = test_column_mapping()
    exit(0 if success else 1)
