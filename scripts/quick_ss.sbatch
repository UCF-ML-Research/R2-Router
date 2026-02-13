#!/bin/bash
#SBATCH -p hpg-default
#SBATCH -c 16
#SBATCH --mem=32gb
#SBATCH -t 02:00:00
#SBATCH -J quick-ss
#SBATCH -o logs/quick_ss_%j.log
#SBATCH --exclude=c1000a-s15
cd /home/ah872032.ucf/jiaqi/router

echo "=== Quick Semi-Supervised Routing ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

.venv/bin/python experiments/quick_semisupervised.py

echo ""
echo "=== Done ==="
echo "Date: $(date)"
