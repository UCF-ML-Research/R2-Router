#!/bin/bash
#SBATCH -p hpg-default
#SBATCH -c 16
#SBATCH --mem=32gb
#SBATCH -t 02:00:00
#SBATCH -J ss-route
#SBATCH -o logs/semisupervised_%j.log
#SBATCH --exclude=c1000a-s15
cd /home/ah872032.ucf/jiaqi/router

echo "=== Semi-Supervised Routing Experiments ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

.venv/bin/python experiments/semisupervised_routing.py

echo ""
echo "=== Done ==="
echo "Date: $(date)"
