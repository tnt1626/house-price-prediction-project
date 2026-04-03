"""
Utility module for core functionality.
Contains MLflow cache management and logging utilities.
"""

import json
import logging
import pickle
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mlflow


class Logger:
    """Professional logging configuration"""

    def __init__(self, name: str, log_file: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        
        if not self.logger.handlers:
            # Console handler with UTF-8 encoding for Windows compatibility
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
            self.logger.setLevel(logging.INFO)
            
            # File handler if specified
            if log_file:
                try:
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                    file_handler = logging.FileHandler(log_file, encoding='utf-8')
                    file_handler.setLevel(logging.DEBUG)
                    file_handler.setFormatter(formatter)
                    self.logger.addHandler(file_handler)
                except (OSError, PermissionError) as e:
                    console_handler.setLevel(logging.WARNING)
                    self.logger.warning(f"Could not create log file {log_file}: {e}. Logging to console only.")

    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message"""
        self.logger.critical(message)


class MLflowTrainingCacheManager:
    """
    Enhanced caching system for training and evaluation results with MLflow integration.
    Thread-safe cache management for ML pipeline results.
    """

    def __init__(self, base_path: Path = Path("training_cache"), logger: Optional[Logger] = None):
        """
        Initialize cache manager.
        
        Parameters:
            base_path: Directory path for cache storage
            logger: Logger instance for tracking operations
        """
        self.base_path = Path(base_path)
        self.cache_lock = threading.Lock()
        self.logger = logger or Logger(__name__)

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist"""
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Could not create cache directory: {e}")

    def save_results(
        self, 
        results: Any, 
        filename: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save training/evaluation results to cache with MLflow logging.
        
        Parameters:
            results: Results object to cache
            filename: Name for cache file (without extension)
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        with self.cache_lock:
            try:
                self._ensure_cache_dir()

                cache_file = self.base_path / f"{filename}.pkl"
                meta_file = self.base_path / f"{filename}_meta.json"

                # Save results
                with open(cache_file, 'wb') as f:
                    pickle.dump(results, f)

                # Save metadata
                if metadata is None:
                    metadata = {}
                metadata.update({
                    'timestamp': datetime.now().isoformat(),
                    'filename': filename
                })

                with open(meta_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

                # Log to MLflow if active
                if mlflow.active_run():
                    mlflow.log_artifact(str(cache_file), "cache")
                    mlflow.log_artifact(str(meta_file), "cache_metadata")

                self.logger.info(f"[OK] Cached results: {filename}")
                return True
                
            except Exception as e:
                self.logger.error(f"✗ Cache save failed: {e}")
                return False

    def load_results(self, filename: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Load training/evaluation results from cache.
        
        Parameters:
            filename: Name of cache file (without extension)
            
        Returns:
            Tuple of (results, metadata) or (None, None) if not found
        """
        with self.cache_lock:
            try:
                cache_file = self.base_path / f"{filename}.pkl"
                meta_file = self.base_path / f"{filename}_meta.json"

                if not cache_file.exists():
                    self.logger.debug(f"Cache file not found: {filename}")
                    return None, None

                # Load results
                with open(cache_file, 'rb') as f:
                    results = pickle.load(f)

                # Load metadata
                metadata = None
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)

                # Log cache hit
                if mlflow.active_run():
                    mlflow.log_param("cache_hit", True)
                    if metadata:
                        mlflow.log_param("cache_timestamp", metadata.get('timestamp', 'unknown'))

                self.logger.info(f"[OK] Loaded cached results: {filename}")
                return results, metadata
                
            except Exception as e:
                self.logger.error(f"✗ Cache load failed: {e}")
                return None, None

    def results_exist(self, filename: str) -> bool:
        """
        Check if cached results exist.
        
        Parameters:
            filename: Name of cache file (without extension)
            
        Returns:
            True if cache file exists, False otherwise
        """
        cache_file = self.base_path / f"{filename}.pkl"
        return cache_file.exists()


def log_to_mlflow(params_dict: Dict[str, Any]) -> None:
    """
    Safely log parameters to MLflow if a run is active.
    
    Parameters:
        params_dict: Dictionary of parameters to log
    """
    if mlflow.active_run():
        try:
            mlflow.log_params(params_dict)
        except Exception as e:
            logging.error(f"Failed to log to MLflow: {e}")
