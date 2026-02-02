"""
Main experimental package for R2-Router.

This package contains all the core functionality for training and evaluating
the R2-Router routing system and baseline methods in IID (in-distribution) setting.

Subpackages:
- core: R2-Router predictor implementations
  - predictor: PyTorch-based neural network predictor
  - predictor_sklearn: Sklearn-based linear regression predictor
  - train_r2: Training script for R2-Router predictors

- baselines: Baseline routing methods
  - carrot: CARROT baseline methods (KNN and Linear)
    - baselines_carrot: Implementation
    - train_carrot: Training script
  - irt: IRT baseline methods (MIRT and NIRT)
    - baselines_irt: Implementation
    - train_irt: Training script

- shared: Shared utilities and data management
  - dataset_manager: Centralized train/test split management
  - llm_loader: Load LLM data and predictor checkpoints
  - router_dataset: Dataset wrapper for training
  - utils: Utility functions

- evaluation: Evaluation and comparison scripts
  - compare_methods: Method comparison script
  - results: Main IID evaluation script
"""

__version__ = "2.0.0"
