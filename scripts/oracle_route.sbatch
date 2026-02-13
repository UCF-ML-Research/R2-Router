#!/bin/bash
#SBATCH -p hpg-default
#SBATCH -c 8
#SBATCH --mem=16gb
#SBATCH -t 01:00:00
#SBATCH -J oracle
#SBATCH -o logs/oracle_%j.log
cd /home/ah872032.ucf/jiaqi/router
.venv/bin/python scripts/oracle_analysis.py
