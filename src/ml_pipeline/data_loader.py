"""
Data loading module for the ML pipeline.
Handles downloading and loading training/test data.
"""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import gdown
import mlflow

from src.core.config import (
    DATA_DIR, 
    TRAIN_DATA_FILE, 
    TEST_SIZE, 
    RANDOM_STATE,
    DATA_URL
)
from src.core.utils import Logger


logger = Logger(__name__)


def download_dataset_from_drive(
    url: str = DATA_URL,
    output_path: Path = TRAIN_DATA_FILE
) -> bool:
    """
    Download dataset from Google Drive using gdown.
    
    Parameters:
        url: Google Drive URL or file ID
        output_path: Path to save downloaded file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if output_path.exists():
            logger.info(f"Dataset already exists at {output_path}")
            return True
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading dataset from Google Drive...")
        
        gdown.download(url, str(output_path), quiet=False)
        
        if output_path.exists():
            logger.info(f"[OK] Dataset downloaded successfully to {output_path}")
            return True
        else:
            logger.error("Downloaded file not found")
            return False
            
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False


def load_data(data_path: Path = TRAIN_DATA_FILE) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Parameters:
        data_path: Path to CSV data file
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"[OK] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Log to MLflow if active
        if mlflow.active_run():
            mlflow.log_params({
                "dataset_rows": df.shape[0],
                "dataset_columns": df.shape[1]
            })
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform initial data cleaning.
    
    Parameters:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Remove ID column as it doesn't contribute to model learning
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        logger.info("[OK] Removed ID column")
    
    return df


def split_train_test(
    df: pd.DataFrame,
    target_column: str = "SalePrice",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Parameters:
        df: Input DataFrame
        target_column: Name of target column
        test_size: Proportion of test set (0.0-1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    X = df.drop(target_column, axis=1)
    y = df[target_column].astype(float)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state
    )
    
    logger.info(f"[OK] Data split: train={len(X_train)}, test={len(X_test)}")
    
    if mlflow.active_run():
        mlflow.log_params({
            "train_size": len(X_train),
            "test_size": len(X_test),
            "test_split_ratio": test_size
        })
        mlflow.log_metrics({
            "train_mean": y_train.mean(),
            "test_mean": y_test.mean(),
            "train_std": y_train.std()
        })
    
    return X_train, X_test, y_train, y_test


def prepare_data(
    data_path: Optional[Path] = None,
    download_if_missing: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Complete data preparation pipeline.
    
    Parameters:
        data_path: Path to data file (uses default if None)
        download_if_missing: Whether to download data if not found
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if data_path is None:
        data_path = TRAIN_DATA_FILE
    
    # Download if needed
    if not data_path.exists() and download_if_missing:
        logger.info("Dataset not found locally, downloading...")
        if not download_dataset_from_drive():
            raise RuntimeError("Failed to download dataset")
    
    # Load data
    df = load_data(data_path)
    
    # Clean data
    df = clean_raw_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_train_test(df)
    
    logger.info("[OK] Data preparation complete")
    return X_train, X_test, y_train, y_test
