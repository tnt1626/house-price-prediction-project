# 🗺️ Navigation & Documentation Index

Welcome! This file helps you navigate the extracted house price prediction project.

---

## 📖 Start Here

### For First-Time Users
1. **[EXTRACTION_SUMMARY.md](EXTRACTION_SUMMARY.md)** ← Read this first!
   - Overview of what was extracted
   - Quick start in 3 steps
   - File descriptions

2. **[training/README.md](training/README.md)** ← Then read this
   - Detailed setup instructions
   - Configuration options
   - Troubleshooting guide

3. **[training/TIPS_AND_TRICKS.md](training/TIPS_AND_TRICKS.md)** ← Advanced topics
   - Performance optimization
   - Development tricks
   - Debugging and profiling

---

## 🚀 Quick Commands

### Install & Run (< 5 minutes)
```bash
cd training

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py

# Make predictions
python inference.py ../data/test.csv
```

### Guided Setup
```bash
cd training
python quickstart.py  # Interactive setup wizard
```

---

## 📁 Project Structure

```
house-price-prediction-project/
│
├── 📄 EXTRACTION_SUMMARY.md          ← Overview of all files
├── 📄 README.md                       ← Original project README
│
├── 📂 training/                       ⭐ Main work directory
│   ├── 🚀 train.py                   ← Main training script
│   ├── 🔮 inference.py               ← Predictions script
│   ├── 🔧 tune_hyperparameters.py    ← Hyperparameter tuning
│   ├── ⚡ quickstart.py              ← Setup wizard
│   ├── 📖 README.md                  ← Detailed documentation
│   ├── 💡 TIPS_AND_TRICKS.md         ← Advanced guide
│   ├── 📋 requirements.txt           ← Dependencies
│   │
│   ├── 📂 modules/                   ← Python package
│   │   ├── __init__.py
│   │   ├── ⚙️ config.py              ← Constants & paths
│   │   ├── 💾 cache_manager.py       ← Result caching
│   │   ├── 🔀 transformers.py        ← Data transforms
│   │   ├── 🎯 feature_engineering.py ← Features
│   │   └── 🤖 models.py              ← Training functions
│   │
│   └── 📂 outputs/                   ← Auto-created
│       ├── logs/                     ← Results & plots
│       ├── training_cache/           ← Cached data
│       └── mlruns/                   ← MLflow tracking
│
├── 📂 notebooks/                      ← Original Jupyter notebooks
│   └── House_price_prediction_project_by_Danh.ipynb
│
├── 📂 data/                           ← Training data
│   └── train.csv
│
└── 📂 src/                            ← Other project files
```

---

## 🎯 Task-Based Navigation

### I want to...

#### **Train models locally**
→ Read: [training/README.md](training/README.md#quick-start)
→ Run: `python training/train.py`

#### **Make predictions**
→ Read: [training/README.md](training/README.md#inference)
→ Run: `python training/inference.py data/test.csv`

#### **Optimize hyperparameters**
→ Read: [training/TIPS_AND_TRICKS.md](training/TIPS_AND_TRICKS.md#advanced-techniques)
→ Run: `python training/tune_hyperparameters.py`

#### **Understand the code**
→ Read: [EXTRACTION_SUMMARY.md](EXTRACTION_SUMMARY.md#-file-descriptions)
→ Browse: `training/modules/`

#### **Speed up training**
→ Read: [training/TIPS_AND_TRICKS.md](training/TIPS_AND_TRICKS.md#-performance-optimization)

#### **Debug errors**
→ Read: [training/README.md](training/README.md#troubleshooting)
→ Check: [training/TIPS_AND_TRICKS.md](training/TIPS_AND_TRICKS.md#-debugging--logging)

#### **Customize the pipeline**
→ Edit: `training/modules/config.py`
→ Reference: [EXTRACTION_SUMMARY.md](EXTRACTION_SUMMARY.md#configuration-module)

#### **Integrate with my app**
→ Read: [training/TIPS_AND_TRICKS.md](training/TIPS_AND_TRICKS.md#-integration-tips)

---

## 📚 Module Reference

### Core Modules

| Module | Purpose | Key Functions |
|--------|---------|--------------|
| **config.py** | Configuration | Constants, paths, hyperparameters |
| **transformers.py** | Data transforms | OrdinalMapper, RarePooler, TargetEncoder |
| **feature_engineering.py** | Feature creation | add_domain_features, make_feature_space |
| **models.py** | Model training | base_models_dict, evaluate_all_models |
| **cache_manager.py** | Result caching | save_results, load_results |

### Main Scripts

| Script | Purpose | Entry Point |
|--------|---------|------------|
| **train.py** | Main pipeline | `python train.py` |
| **inference.py** | Predictions | `python inference.py <data>` |
| **tune_hyperparameters.py** | Optuna tuning | Import and call functions |
| **quickstart.py** | Setup wizard | `python quickstart.py` |

---

## 🔗 Key Sections

### Configuration & Setup
- [Config Options](training/README.md#-configuration)
- [Dependencies](training/README.md#-installation-required)
- [Data Format](training/README.md#-prepare-data)

### Usage & Examples
- [Quick Start](training/README.md#-quick-start)
- [Training Examples](training/README.md#-training-pipeline-flow)
- [Customization Examples](training/README.md#-customization-examples)

### Advanced Topics
- [Performance Optimization](training/TIPS_AND_TRICKS.md#-performance-optimization)
- [GPU Acceleration](training/TIPS_AND_TRICKS.md#2-enable-gpu-acceleration)
- [Ensemble Methods](training/TIPS_AND_TRICKS.md#1-ensemble-methods)
- [Feature Selection](training/TIPS_AND_TRICKS.md#3-automated-feature-selection)

### Troubleshooting
- [Common Errors](training/README.md#troubleshooting)
- [Memory Issues](training/TIPS_AND_TRICKS.md#-performance-optimization)
- [Debugging Tips](training/TIPS_AND_TRICKS.md#-debugging--logging)

---

## 📊 Example Workflows

### Workflow 1: First-Time User
```
1. Read EXTRACTION_SUMMARY.md (2 min)
2. Run quickstart.py (5 min)
3. Read training/README.md (10 min)
4. Run python training/train.py (10-15 min)
5. Make predictions: python training/inference.py data/test.csv (1 min)
```

### Workflow 2: Developer
```
1. Read EXTRACTION_SUMMARY.md
2. Review training/modules/ structure
3. Edit training/modules/config.py (customize)
4. Run python training/train.py (with custom config)
5. Iterate based on results
```

### Workflow 3: Optimization Expert
```
1. Run baseline: python training/train.py
2. Review results in training/outputs/logs/
3. Read TIPS_AND_TRICKS.md
4. Run python training/tune_hyperparameters.py
5. Compare improvements
```

### Workflow 4: API Integration
```
1. Understand module structure
2. Load model: from modules import MLflowTrainingCacheManager
3. Create FastAPI wrapper (see TIPS_AND_TRICKS.md)
4. Deploy as REST API
```

---

## 📝 File Checklist

All extracted files:

Core Scripts:
- ✅ `training/train.py` - Main training
- ✅ `training/inference.py` - Predictions
- ✅ `training/tune_hyperparameters.py` - Hyperparameter tuning
- ✅ `training/quickstart.py` - Setup wizard

Modules:
- ✅ `training/modules/__init__.py`
- ✅ `training/modules/config.py`
- ✅ `training/modules/cache_manager.py`
- ✅ `training/modules/transformers.py`
- ✅ `training/modules/feature_engineering.py`
- ✅ `training/modules/models.py`

Documentation:
- ✅ `EXTRACTION_SUMMARY.md`
- ✅ `training/README.md`
- ✅ `training/TIPS_AND_TRICKS.md`
- ✅ This file (navigation index)

Configuration:
- ✅ `training/requirements.txt`

---

## 🆘 Getting Help

### I'm stuck on...

**Installation?**
→ See [training/README.md - Troubleshooting](training/README.md#troubleshooting)

**Running train.py?**
→ Try `python training/quickstart.py` first

**Understanding the code?**
→ Start with [EXTRACTION_SUMMARY.md](EXTRACTION_SUMMARY.md)

**Customizing models?**
→ Check [TIPS_AND_TRICKS.md - Development Tricks](training/TIPS_AND_TRICKS.md#-development-tricks)

**Performance issues?**
→ Read [TIPS_AND_TRICKS.md - Performance Optimization](training/TIPS_AND_TRICKS.md#-performance-optimization)

---

## 🎓 Learning Path

### Beginner
1. Read EXTRACTION_SUMMARY.md
2. Run quickstart.py
3. Review training/README.md
4. Run train.py successfully

### Intermediate
1. Study training/modules/ code
2. Read TIPS_AND_TRICKS.md
3. Modify config.py and retrain
4. Analyze results

### Advanced
1. Implement custom transformers
2. Add new model types
3. Create ensemble models
4. Deploy as API

---

## 📞 Quick Reference

**Change random seed:**
Edit `training/modules/config.py`: `RANDOM_STATE = 42`

**Use only specific models:**
Edit `training/modules/models.py` in `base_models_dict()`

**Adjust test split:**
Edit `training/train.py`: `test_size = 0.2`

**Enable MLflow:**
Edit `training/modules/config.py`: `USE_MLFLOW = True`

**Reduce training time:**
See [TIPS_AND_TRICKS.md - Reduce Training Time](training/TIPS_AND_TRICKS.md#1-reduce-training-time)

---

## ✨ You're All Set!

Everything is ready to train your house price prediction model locally. 

**Start here:** `cd training && python quickstart.py`

Happy training! 🚀

---

**Questions?** Check the documentation files first - they contain answers to most common issues!
