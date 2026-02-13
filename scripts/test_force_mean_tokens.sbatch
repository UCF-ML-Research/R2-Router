#!/bin/bash
#SBATCH -p hpg-default
#SBATCH -c 8
#SBATCH --mem=16gb
#SBATCH -t 00:30:00
#SBATCH -J meantoken
#SBATCH -o logs/meantoken_%j.log
cd /home/ah872032.ucf/jiaqi/router

echo "=== Test: force-mean-tokens (token pred = category mean always) ==="
echo "Using checkpoint from 10% KNN training"
echo ""

# Compare with and without --force-mean-tokens
for lam in 0.0 0.5 0.999; do
  for sk in 0 4; do
    echo "--- lambda=$lam sk=$sk, 4 models, 9 budgets, force-mean-tokens ---"
    .venv/bin/python scripts/route_and_eval.py --lambda_val $lam --shrinkage_k $sk \
        --exclude-budgets budget_unlimited budget_1500 \
        --models 235b 80b gemini-flash haiku \
        --force-mean-tokens 2>&1 | grep "sweep files"
    echo ""
  done
done

echo "=== Concise-only ==="
echo ""
for lam in 0.0 0.5 0.999; do
  for sk in 0 4; do
    echo "--- concise-only: lambda=$lam sk=$sk, force-mean-tokens ---"
    .venv/bin/python scripts/route_and_eval.py --lambda_val $lam --shrinkage_k $sk \
        --exclude-budgets budget_10 budget_20 budget_40 budget_80 budget_150 budget_200 budget_400 budget_800 budget_1500 budget_unlimited \
        --models 235b 80b gemini-flash haiku \
        --force-mean-tokens 2>&1 | grep "sweep files"
    echo ""
  done
done
