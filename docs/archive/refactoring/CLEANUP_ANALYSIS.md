# Cleanup Analysis: Active vs Legacy Files

This document identifies which files are actively used vs obsolete/redundant in the LLM routing codebase.

---

## Python Code Files Analysis

### ✅ ACTIVE - Currently Used by Pipeline

**Core Infrastructure (Required):**
- `dataset_manager.py` - Centralized train/test split management
- `router_dataset.py` - Dataset wrapper for train/test subsets
- `llm_loader.py` - Loads individual LLM data with checkpoints
- `utils.py` - Utility functions (JSON parsing, plotting, data loading)

**Predictors:**
- `predictor_sklearn.py` - **PRIMARY**: Sklearn-based predictors (Ridge, Lasso, RF, GBM, MLP)
  - Used by: `train_core.py`, `compare_methods.py`
- `predictor.py` - **SECONDARY**: PyTorch neural networks + `route_scores()` function
  - Used by: `results.py`, `baselines_irt.py`, some routing logic

**Baselines:**
- `baselines_carrot.py` - CARROT baselines (KNN + Linear)
  - Used by: `train_carrot.py`, `compare_methods.py`, `results.py`
- `baselines_irt.py` - IRT-based baselines (optional)
  - Used by: `results.py`, OOD evaluation

**Entry Points:**
- `train_core.py` - Trains CoRE predictors (called by run_experiment.sh)
- `train_carrot.py` - Trains CARROT baselines (called by run_experiment.sh)
- `compare_methods.py` - Compares methods (called by run_experiment.sh)
- `results.py` - Manual evaluation script (legacy, not called by run_experiment.sh)

**Utilities:**
- `check_data.py` - Data validation tool (standalone utility)

**Orchestrator:**
- `run_experiment.sh` - Main pipeline script

**Total Active: 14 files**

---

### 🗑️ LEGACY - Obsolete/Redundant (Can Delete)

**Superseded by Split Files:**
- ❌ `baselines.py` (446 lines)
  - Monolithic version containing both CARROT and IRT baselines
  - Split into: `baselines_carrot.py` + `baselines_irt.py`
  - Not imported anywhere
  - **Recommendation: DELETE**

**Superseded by Better Version:**
- ❌ `compare_core_carrot.py` (450 lines)
  - Older comparison script with hardcoded model list
  - Replaced by: `compare_methods.py` (more flexible, command-line args)
  - Not called by run_experiment.sh
  - **Recommendation: DELETE**

**Test Files (Old Refactoring Tests):**
- ❌ `test_sklearn_refactor.py`
  - Tests old sklearn predictor refactoring
  - Refactoring complete, test no longer needed
  - **Recommendation: DELETE**

- ❌ `test_refactor_api.py`
  - Tests old API signature changes
  - API stable, test no longer needed
  - **Recommendation: DELETE**

**Potentially Useful Test:**
- ⚠️ `test_baselines_split.py`
  - Tests baseline implementation correctness
  - Not run automatically, but could be useful for validation
  - **Recommendation: KEEP or MOVE to tests/ folder**

**Total Legacy: 4-5 files safe to delete**

---

### 📂 OOD Evaluation (Separate Pipeline)

**Active OOD Files:**
- `ood_evaluation/ood_dataset_manager.py`
- `ood_evaluation/run_ood.py`
- `ood_evaluation/map_and_split_data.py`
- `ood_evaluation/README.md`

**Status:** Independent pipeline, keep if needed, can ignore if not using OOD

**Archive:**
- `ood_evaluation/archive/*` (7 old scripts)
- **Recommendation: DELETE archive folder**

---

## Documentation Files Analysis

### ✅ CURRENT USER-FACING DOCS (Keep)

**Primary Documentation:**
1. **`CLAUDE.md`** (22K) - **ESSENTIAL**
   - Project overview and architecture
   - Used by Claude Code for AI assistance
   - Contains core concepts and file descriptions
   - **Status: REQUIRED - Keep updated**

2. **`README_EXPERIMENT.md`** (9.3K) - **ESSENTIAL**
   - Main user guide for run_experiment.sh
   - How to configure LLM pool and hyperparameters
   - Examples and usage instructions
   - **Status: REQUIRED - Primary user documentation**

3. **`CHECKPOINT_MANAGEMENT.md`** (5.8K) - **IMPORTANT**
   - Explains CoRE vs CARROT checkpoint behavior
   - Configuration tracking logic
   - Created today, highly relevant
   - **Status: KEEP - Important reference**

4. **`HYPERPARAMETER_TUNING.md`** (6.8K) - **IMPORTANT**
   - Guide for tuning predictor_sklearn.py hyperparameters
   - Model selection strategies
   - Troubleshooting guide
   - **Status: KEEP - User reference**

5. **`EXPERIMENTAL_SETTINGS.md`** (8.8K) - **USEFUL**
   - Dataset description
   - Experimental setup details
   - Baseline configurations
   - **Status: KEEP - Research documentation**

**Total Current Docs: 5 files**

---

### 🗑️ HISTORICAL/REFACTORING DOCS (Can Archive)

These document past refactoring work. Useful for history but not for daily use:

**Refactoring History (Completed Work):**
1. ❌ `BASELINE_REFACTORING_SUMMARY.md` (8.6K)
   - Documents old baseline refactoring
   - Work completed, superseded by current code
   - **Recommendation: ARCHIVE or DELETE**

2. ❌ `BASELINE_SPLIT_SUMMARY.md` (12K)
   - Documents splitting baselines.py into separate files
   - Work completed
   - **Recommendation: ARCHIVE or DELETE**

3. ❌ `CARROT_REFACTORING_SUMMARY.md` (8.8K)
   - Documents CARROT baseline extraction
   - Work completed
   - **Recommendation: ARCHIVE or DELETE**

4. ❌ `REFACTORING_SUMMARY.md` (6.7K)
   - General refactoring documentation
   - Old work
   - **Recommendation: ARCHIVE or DELETE**

5. ❌ `SKLEARN_REFACTORING_SUMMARY.md` (7.3K)
   - Documents sklearn predictor refactoring
   - Work completed
   - **Recommendation: ARCHIVE or DELETE**

6. ❌ `THREE_COMPONENT_REFACTORING.md` (6.8K)
   - Documents three-component architecture split
   - Work completed
   - **Recommendation: ARCHIVE or DELETE**

**API Change History (Completed Work):**
7. ❌ `FINAL_API_SUMMARY.md` (12K)
   - Final API documentation after refactoring
   - Superseded by current code
   - **Recommendation: ARCHIVE or DELETE**

8. ❌ `NEW_API_SUMMARY.md` (8.7K)
   - Documents API changes (separate limited/unlimited)
   - Work completed
   - **Recommendation: ARCHIVE or DELETE**

9. ❌ `NAMING_IMPROVEMENTS.md` (5.9K)
   - Documents parameter naming changes
   - Work completed
   - **Recommendation: ARCHIVE or DELETE**

**Architecture Documentation (Potentially Redundant):**
10. ⚠️ `PREDICTOR_ARCHITECTURE.md` (8.9K)
    - Documents predictor architecture in detail
    - Some overlap with CLAUDE.md
    - **Recommendation: MERGE into CLAUDE.md or KEEP as separate reference**

11. ⚠️ `PREDICTOR_GUIDE.md` (5.0K)
    - How to switch between PyTorch/sklearn predictors
    - Still somewhat relevant
    - **Recommendation: MERGE into README_EXPERIMENT.md or KEEP**

**Total Historical Docs: 9 can archive, 2 need decision**

---

## Recommended Actions

### Immediate Cleanup (High Confidence)

**Delete These Python Files:**
```bash
rm baselines.py
rm compare_core_carrot.py
rm test_sklearn_refactor.py
rm test_refactor_api.py
rm -rf ood_evaluation/archive/
```

**Move to Archive Folder (Refactoring Docs):**
```bash
mkdir -p docs/archive/refactoring
mv BASELINE_REFACTORING_SUMMARY.md docs/archive/refactoring/
mv BASELINE_SPLIT_SUMMARY.md docs/archive/refactoring/
mv CARROT_REFACTORING_SUMMARY.md docs/archive/refactoring/
mv REFACTORING_SUMMARY.md docs/archive/refactoring/
mv SKLEARN_REFACTORING_SUMMARY.md docs/archive/refactoring/
mv THREE_COMPONENT_REFACTORING.md docs/archive/refactoring/
mv FINAL_API_SUMMARY.md docs/archive/refactoring/
mv NEW_API_SUMMARY.md docs/archive/refactoring/
mv NAMING_IMPROVEMENTS.md docs/archive/refactoring/
```

---

### Files That Should Remain

**Active Python Code (14 files):**
- Core: dataset_manager.py, router_dataset.py, llm_loader.py, utils.py
- Predictors: predictor_sklearn.py, predictor.py
- Baselines: baselines_carrot.py, baselines_irt.py
- Entry points: train_core.py, train_carrot.py, compare_methods.py, results.py
- Utilities: check_data.py
- Orchestrator: run_experiment.sh

**Current Documentation (5 files):**
- CLAUDE.md (AI assistant context)
- README_EXPERIMENT.md (primary user guide)
- CHECKPOINT_MANAGEMENT.md (checkpoint behavior)
- HYPERPARAMETER_TUNING.md (tuning guide)
- EXPERIMENTAL_SETTINGS.md (research documentation)

**Potential Keeps (Need Review):**
- PREDICTOR_ARCHITECTURE.md (detailed architecture - merge or keep?)
- PREDICTOR_GUIDE.md (predictor switching - merge or keep?)
- test_baselines_split.py (validation test - move to tests/ or delete?)

---

## Cleanup Summary

**Before Cleanup:**
- Python files: 18 total
- Markdown docs: 16 total
- Total: 34 files

**After Cleanup:**
- Python files: 14 active
- Markdown docs: 5 essential + 2 optional = 7 max
- Archive: 9 historical docs (moved, not deleted)
- Deleted: 4 obsolete Python files + 1 archive folder
- Total active: ~21 files

**Space Saved:**
- Python: ~5 obsolete files
- Docs: 9 moved to archive
- Cleaner workspace with only relevant files visible

---

## Decision Matrix

| File | Type | Status | Action | Reason |
|------|------|--------|--------|--------|
| baselines.py | Code | OBSOLETE | DELETE | Superseded by split files |
| compare_core_carrot.py | Code | OBSOLETE | DELETE | Superseded by compare_methods.py |
| test_sklearn_refactor.py | Test | OBSOLETE | DELETE | Old test, refactoring done |
| test_refactor_api.py | Test | OBSOLETE | DELETE | Old test, API stable |
| test_baselines_split.py | Test | USEFUL | KEEP/MOVE | Could validate baselines |
| 9× refactoring docs | Docs | HISTORICAL | ARCHIVE | Past work, not daily use |
| PREDICTOR_ARCHITECTURE.md | Docs | USEFUL | KEEP/MERGE | Detailed reference |
| PREDICTOR_GUIDE.md | Docs | USEFUL | KEEP/MERGE | Switching guide |
| CLAUDE.md | Docs | ESSENTIAL | KEEP | AI context |
| README_EXPERIMENT.md | Docs | ESSENTIAL | KEEP | Primary guide |
| CHECKPOINT_MANAGEMENT.md | Docs | ESSENTIAL | KEEP | Important reference |
| HYPERPARAMETER_TUNING.md | Docs | ESSENTIAL | KEEP | User guide |
| EXPERIMENTAL_SETTINGS.md | Docs | USEFUL | KEEP | Research docs |

---

## Next Steps

1. **Review this analysis** - Do you agree with these categorizations?

2. **Create archive folder** if you want to preserve historical docs:
   ```bash
   mkdir -p docs/archive/refactoring
   ```

3. **Execute cleanup** - Run the commands in "Immediate Cleanup" section

4. **Update CLAUDE.md** - Remove references to deleted files

5. **Consider merging** PREDICTOR_ARCHITECTURE.md and PREDICTOR_GUIDE.md into existing docs

Would you like me to proceed with any of these cleanup actions?
