#!/bin/bash
#SBATCH -p hpg-default
#SBATCH -c 8
#SBATCH --mem=16gb
#SBATCH -t 01:00:00
#SBATCH -J baseline10
#SBATCH -o logs/baseline10_%j.log
cd /home/ah872032.ucf/jiaqi/router

echo "=== Baseline comparison: Ridge 10% train, original params ==="
echo "Training with --test-size 0.9 (10% train, 90% test)..."
echo ""

# Train Ridge with 10% data (same as original baseline method)
.venv/bin/python scripts/train_category_predictors.py \
    --ridge-only --test-size 0.9

echo ""
echo "=== Evaluating routing with original baseline params ==="
echo ""

# Original baseline: lambda=0.999, shrinkage_k=4, 4 models, 9 budgets
echo "--- Original baseline: lambda=0.999 shrinkage_k=4, 4 models, 9 budgets ---"
.venv/bin/python scripts/route_and_eval.py --lambda_val 0.999 --shrinkage_k 4 \
    --exclude-budgets budget_unlimited budget_1500 \
    --models 235b 80b gemini-flash haiku

echo ""
# Also test a few other configs for comparison
for lam in 0.0 0.5 0.999; do
  for sk in 0 4; do
    echo "--- lambda=$lam shrinkage_k=$sk, 4 models, 9 budgets ---"
    .venv/bin/python scripts/route_and_eval.py --lambda_val $lam --shrinkage_k $sk \
        --exclude-budgets budget_unlimited budget_1500 \
        --models 235b 80b gemini-flash haiku
    echo ""
  done
done

echo ""
echo "=== Also test concise-only for comparison ==="
echo ""
for lam in 0.0 0.999; do
  for sk in 0 4; do
    echo "--- concise-only: lambda=$lam shrinkage_k=$sk, 4 models ---"
    .venv/bin/python scripts/route_and_eval.py --lambda_val $lam --shrinkage_k $sk \
        --exclude-budgets budget_10 budget_20 budget_40 budget_80 budget_150 budget_200 budget_400 budget_800 budget_1500 budget_unlimited \
        --models 235b 80b gemini-flash haiku
    echo ""
  done
done
