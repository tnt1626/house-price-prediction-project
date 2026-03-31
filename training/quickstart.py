#!/usr/bin/env python3
# ============================================================================
# Quick Start Script - Set up and run training
# ============================================================================
import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(num, text):
    """Print formatted step"""
    print(f"\n[Step {num}] {text}")


def check_dependencies():
    """Check if required packages are installed"""
    
    print_step(1, "Checking dependencies...")
    
    required = ['pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn']
    optional = ['xgboost', 'catboost', 'lightgbm', 'optuna', 'mlflow']
    
    missing_required = []
    missing_optional = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg}")
            missing_required.append(pkg)
    
    print("\n  Optional packages:")
    for pkg in optional:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ⚠️  {pkg} (optional)")
            missing_optional.append(pkg)
    
    if missing_required:
        print_header("INSTALLATION REQUIRED")
        print("\nRun the following to install missing packages:")
        print(f"  pip install {' '.join(missing_required)}")
        print("\nOr install all dependencies:")
        print("  pip install -r requirements.txt")
        return False
    
    return True


def check_data():
    """Check if training data exists"""
    
    print_step(2, "Checking data...")
    
    # Expected data paths
    paths_to_check = [
        Path("../data/train-house-prices-advanced-regression-techniques.csv"),
        Path("./data/train-house-prices-advanced-regression-techniques.csv"),
        Path("../../data/train-house-prices-advanced-regression-techniques.csv"),
    ]
    
    for path in paths_to_check:
        if path.exists():
            print(f"  ✅ Found data: {path.resolve()}")
            return str(path)
    
    print("  ⚠️  Data file not found")
    print("\n  Expected locations:")
    for path in paths_to_check:
        print(f"    - {path}")
    
    print("\n  Please ensure the CSV file is in one of these locations:")
    print("  - ../data/")
    print("  - ./data/")
    
    return None


def create_directories():
    """Create necessary directories"""
    
    print_step(3, "Creating directories...")
    
    dirs = [
        "outputs",
        "outputs/logs",
        "outputs/training_cache",
        "outputs/mlruns",
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_path}")


def run_training():
    """Run the training script"""
    
    print_step(4, "Starting training...")
    print("\n  This may take 5-15 minutes depending on your hardware.")
    print("  Models will be evaluated and results cached.\n")
    
    try:
        # Run training
        result = subprocess.run(
            [sys.executable, "train.py"],
            cwd=Path(__file__).parent,
            capture_output=False
        )
        
        if result.returncode == 0:
            print_header("✅ TRAINING COMPLETE!")
            print("\n📊 Results saved to:")
            print("  - outputs/logs/baseline_results.csv")
            print("  - outputs/logs/final_house_price_model.pkl")
            print("\n📈 Next steps:")
            print("  1. View results: python inference.py ../data/test-data.csv")
            print("  2. Tune hyperparameters: python tune_hyperparameters.py")
            return True
        else:
            print_header("❌ TRAINING FAILED")
            print("Check the error messages above.")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running training: {e}")
        return False


def main():
    """Main quick start flow"""
    
    print_header("🏠 HOUSE PRICE PREDICTION - QUICK START")
    
    # Change to training directory
    training_dir = Path(__file__).parent
    os.chdir(training_dir)
    
    print(f"\nWorking directory: {training_dir.resolve()}\n")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n⚠️  Please install missing packages and try again.")
        sys.exit(1)
    
    # Step 2: Check data
    data_path = check_data()
    if not data_path:
        print("\n⚠️  Please ensure training data is available.")
        print("    Download from: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques")
        sys.exit(1)
    
    # Step 3: Create directories
    create_directories()
    
    # Step 4: Run training
    if not run_training():
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("  🎉 All done! Happy predicting!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
