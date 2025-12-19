# Test Scripts

This directory contains validation and test scripts for the CoRE Router project.

## Test Files

### `test_baselines_split.py`
**Purpose**: Validates the split baseline files work correctly after refactoring

**Tests**:
- CARROT-KNN baseline (save/load/predict)
- CARROT-Linear baseline (save/load/predict)
- IRT (MIRT) baseline (save/load/predict)
- NIRT baseline (save/load/predict)
- Backward compatibility (old class names still work)

**Usage**:
```bash
python tests/test_baselines_split.py
```

**Requirements**: Pre-trained checkpoints for GLM-4.5-Air and Llama-3.2-3B

---

### `test_sklearn_refactor.py`
**Purpose**: Validates sklearn predictor refactoring

**Tests**: Sklearn-based `TokenPerformancePredictor` implementation

**Usage**:
```bash
python tests/test_sklearn_refactor.py
```

---

### `test_refactor_api.py`
**Purpose**: Validates API refactoring

**Tests**: Core API functionality after refactoring

**Usage**:
```bash
python tests/test_refactor_api.py
```

---

## Notes

- These are **validation scripts**, not continuous integration tests
- Created during development/refactoring to ensure correctness
- Run them if you modify baseline or predictor implementations
- Some tests may create temporary checkpoints in `./test_checkpoints/` and plots in `./test_plots/`
