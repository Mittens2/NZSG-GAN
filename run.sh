#!/bin/bash
#SBATCH --account=def-hufu
#SBATCH --gres=gpu:1
#SBATCH --mem=4000
#SBATCH --time=0-08:00
#SBATCH --output=%N-%j.out
source /home/amitkad/projects/def-hufu/amitkad/jupyter/bin/activate
python nzsg_app.py
