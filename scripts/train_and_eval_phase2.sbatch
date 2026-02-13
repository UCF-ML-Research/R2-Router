#!/bin/bash
#SBATCH -p hpg-default
#SBATCH -c 8
#SBATCH --mem=16gb
#SBATCH -t 01:00:00
#SBATCH -J phase2
#SBATCH -o logs/phase2_%j.log
cd /home/ah872032.ucf/jiaqi/router

echo "=== Phase 2: LogisticRegression quality predictors ==="
echo "Training with --test-size 0.9 (10% train, 90% test)..."
echo ""

# Train with Logistic, 10% data, Ridge-only (no MLP/KNN search)
.venv/bin/python scripts/train_category_predictors.py \
    --logistic --ridge-only --test-size 0.9

echo ""
echo "=== Evaluating routing ==="
echo ""

# Evaluate routing on full set against sweep GT
# Test different lambda and shrinkage configs

for lam in 0.0 0.5 0.9 0.999; do
  for sk in 0 3; do
    echo "--- lambda=$lam shrinkage_k=$sk ---"

    # Concise only
    .venv/bin/python scripts/route_and_eval.py --lambda_val $lam --shrinkage_k $sk \
        --exclude-budgets budget_10 budget_20 budget_40 budget_80 budget_150 budget_200 budget_400 budget_800 budget_1500 budget_unlimited \
        --models 235b 80b gemini-flash haiku 2>&1 | grep "sweep files"

    # All budgets
    .venv/bin/python scripts/route_and_eval.py --lambda_val $lam --shrinkage_k $sk \
        --models 235b 80b gemini-flash haiku 2>&1 | grep "sweep files"
    echo ""
  done
done

echo ""
echo "=== Also test original Ridge for comparison ==="
echo ""

# Retrain with Ridge (original) at same 10% split
.venv/bin/python scripts/train_category_predictors.py \
    --ridge-only --test-size 0.9

echo ""
for lam in 0.0 0.999; do
  for sk in 0 3; do
    echo "--- Ridge: lambda=$lam shrinkage_k=$sk ---"
    .venv/bin/python scripts/route_and_eval.py --lambda_val $lam --shrinkage_k $sk \
        --exclude-budgets budget_10 budget_20 budget_40 budget_80 budget_150 budget_200 budget_400 budget_800 budget_1500 budget_unlimited \
        --models 235b 80b gemini-flash haiku 2>&1 | grep "sweep files"
    echo ""
  done
done
