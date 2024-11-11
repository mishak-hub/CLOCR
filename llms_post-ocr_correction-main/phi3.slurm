#!/bin/bash
#SBATCH --job-name=phi3        # Job name
#SBATCH --output=phi3.%j.out   # Output file for the job
#SBATCH --error=phi3.%j.err    # Error file for the job
#SBATCH --time=01:00:00          # Maximum runtime (e.g., 2 hours)
#SBATCH --partition=normal       # Partition (queue) name
#SBATCH --ntasks=1               # Number of tasks (1 for sequential execution)
#SBATCH --cpus-per-task=20        # Number of CPU cores per task
#SBATCH --mem=32G                # Memory per node
#SBATCH --gres=gpu:1             # Number of GPUs

# Load required modules (if any)
# module load python/3.9

nvidia-smi

# Activate your conda environment
# source /path/to/your/conda/bin/activate your_environment_name
# source /home/cap6614.student2/miniconda3/activate base
source activate base
conda init
conda activate clocr-gpu

srun python phi3.py

