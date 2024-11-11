#!/bin/bash
#SBATCH --constraint=h100
#SBATCH --job-name=LM_OCR        # Job name
#SBATCH --output=LM_OCR.%j.out   # Output file for the job
#SBATCH --error=LM_OCR.%j.err    # Error file for the job
#SBATCH --time=08:00:00          # Maximum runtime (e.g., 2 hours)
#SBATCH --partition=normal       # Partition (queue) name
#SBATCH --ntasks=1               # Number of tasks (1 for sequential execution)
#SBATCH --cpus-per-task=48        # Number of CPU cores per task
#SBATCH --mem=32G                # Memory per node
#SBATCH --gres=gpu:1             # Number of GPUs

# Load required modules (if any)
# module load python/3.9

nvidia-smi

# Activate your conda environment
module purge
module load cuda/cuda-11.3

module load anaconda
conda env create -f phi3-environment.yml

conda activate phi3-clocr
# huggingface-cli login --token # add token here

module list

# source /path/to/your/conda/bin/activate your_environment_name
# source /home/cap6614.student2/miniconda3/activate base
# source activate base
# conda init
# conda activate clocr-gpu

# Parse command line arguments
FINETUNE=false
PREPARE_DATASET=false
TEST=false
CONFIG_FILE=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --fine_tune) FINETUNE=true ;;
        --prepare_dataset) PREPARE_DATASET=true ;;
        --test) TEST=true ;;
        --config_file)
            CONFIG_FILE="$2"
            shift
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if the configuration file path was provided
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: --config_file is required."
    exit 1
fi


# Step 1: Run the dataset preparation script
if [ "$PREPARE_DATASET" = true ]; then
    echo "Running dataset preparation script with config file: $CONFIG_FILE"
    srun python pipeline.py prepare_datasets "$CONFIG_FILE"
    if [ $? -ne 0 ]; then
        echo "Dataset preparation script failed. Exiting."
        exit 1
    fi
fi

# Step 2: Run the fine_tune script
if [ "$FINETUNE" = true ]; then
    echo "Running fine_tune script with config file: $CONFIG_FILE"
    srun python pipeline.py fine_tune "$CONFIG_FILE"
    if [ $? -ne 0 ]; then
        echo "Finetune script failed. Exiting."
        exit 1
    fi
fi

# Step 3: Run the test script
if [ "$TEST" = true ]; then
    echo "Running test script with config file: $CONFIG_FILE"
    srun python pipeline.py test "$CONFIG_FILE"
    if [ $? -ne 0 ]; then
        echo "Test script failed. Exiting."
        exit 1
    fi
fi


echo "All tasks completed successfully."
