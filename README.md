# CLOCR

## Prerequisites
* Anaconda: [WSL Installation Guide](https://medium.com/hydroinformatics/software-development-in-linux-install-miniconda-in-wsl-27e809a0c064). Newton already has Anaconda installed. 
* Rust/Cargo: Just run the following bash command: `curl https://sh.rustup.rs -sSf | sh` ([source](https://doc.rust-lang.org/cargo/getting-started/installation.html))

## To set up llms_post-ocr_correction-main Folder
This folder hosts the training suite. `ls` into that folder, and create the conda environemnt. Newton doesn't like Conda installing things with multiple threads without being in a slurm job. So, do this:
```bash
srun -p main --time=02:00:00 --ntasks-per-node 2 --gres gpu:1 --pty bash
```
Then, continue through the sections below. If you mess something up and need to restart, and kill the old slurm `bash`, just do this:
```bash
squeue --me
skill <id_of_job_to_kill>
```

```bash
module load anaconda
module load cuda/cuda-11.3
conda env create -f environment.yml 
conda activate clocr-gpu
```
Newton defaultly requires module `anaconda` be loaded, hence its inclusion here. This means we do not have to install Miniconda, as a previous version of this repository required.

## Running the Pipeline
### SLURM script
A model pipeline script is provided for your testing convenience. To use the script, you must provide the `--config_file` argument (read below) and at least one of the task flags:
* `--fine_tune`: Runs the `fine_tune()` task in `pipeline.py` if present.
* `--prepare_dataset`: Runs the `prepare_dataset()` task in `pipeline.py` if present.
* `--test`: Runs the `test()` task in `pipeline.py` if present.

You can use any combination of flags. To run all tasks (recommended):
```bash
sbatch model_pipeline.slurm --config_file path/to/test.yaml --fine_tune --prepare_dataset --test
```

To run only the fine_tune task:
```bash
sbatch model_pipeline.slurm --config_file path/to/test.yaml --fine_tune
```

To run only the prepare_dataset and test tasks:
```bash
sbatch model_pipeline.slurm --config_file path/to/test.yaml --prepare_dataset --test
```

### YAML File
The pipeline uses yaml files to establish model training etc. The `yaml_builder.py` file automagically fills the `models` directory with yaml files and prints to console what `sbatch` commands to run to run the set of 11 tests we have devised for the purposes of our paper. This script could be modified to run tests of your own as well. A simple `python yaml_builder.py` will do the trick in any stock python3 environment. 

Here are the 11 tests we ran:
```bash
sbatch model_pipeline.slurm --config_file models/phi-3-mini-4k_expanded-10k_prompt_8.yaml --fine_tune --test --prepare_dataset
sbatch model_pipeline.slurm --config_file models/phi-3-mini-128k_expanded-10k_prompt_8.yaml --fine_tune --test --prepare_dataset
sbatch model_pipeline.slurm --config_file models/llama-2-7b_llama2-compromise_prompt_1.yaml --fine_tune --test --prepare_dataset
sbatch model_pipeline.slurm --config_file models/llama-2-7b_llama2-compromise_prompt_8.yaml --fine_tune --test --prepare_dataset
sbatch model_pipeline.slurm --config_file models/llama-2-13b_llama2-compromise_prompt_1.yaml --fine_tune --test --prepare_dataset
sbatch model_pipeline.slurm --config_file models/llama-2-13b_llama2-compromise_prompt_8.yaml --fine_tune --test --prepare_dataset
sbatch model_pipeline.slurm --config_file models/bart-base_bart-compromise_prompt_1.yaml --fine_tune --test --prepare_dataset
sbatch model_pipeline.slurm --config_file models/bart-large_bart-compromise_prompt_1.yaml --fine_tune --test --prepare_dataset
sbatch model_pipeline.slurm --config_file models/llama-3.1-7b_base_prompt_1.yaml --fine_tune --test --prepare_dataset
sbatch model_pipeline.slurm --config_file models/llama-3.1-13b_base_prompt_1.yaml --fine_tune --test --prepare_dataset
sbatch model_pipeline.slurm --config_file models/llama-3.1-13b_expanded-10k_prompt_8.yaml --fine_tune --test --prepare_dataset
```

## Datasets

Datasets must be prepared in the `/datasets` folder, as .csv files containing `'OCR Text'`, `'Ground Truth'`, and `'Sample ID'` columns. `'Sample ID'` must be unique for each sequence entry. 

### Example
Here is an example run and yaml file:
```bash
sbatch model_pipeline.slurm --config_file models/bart-base.yaml --fine_tune --prepare_dataset --test
sbatch model_pipeline.slurm --config_file models/bart-large.yaml --fine_tune --prepare_dataset --test
```
models/bart-base.yaml:
```yaml
# Model and Configuration Details
model_type: "BART"               # Specify the model type, e.g., 'BART', 'Llama_2'
config_path: "model_configs.yaml"  # Path to the model's configuration file, local to working dir

# Model Version Details
model_version: "bart-base" # Model version to load from HuggingFace

# Datasets
datasets:                         # List of datasets with their names
  # - "iam_tesseract"
  - "bln600"

# Output Paths
weights_out_path: "bart-base"   # Goes into the `model_weights` folder
statistics_out_path: "bart-base.csv" # Goes into the `results` folder

# Prompt Pattern
prompt_pattern: 1                # Integer representing the selected prompt pattern (see prompts.py)
```

### Dataset preparation
To make a (OCR output,ground truth) corpus goes in resources folder.
each line is text in some corpuses, but for most it is not. Synthdog doesn't care -- it just grabs random sentences from the corpus to make samples from.

`main_synthdog.slurm` generates synthdog files. You give it a config with corpus path yaml and an output path.
`-c`: quantity of images to generate

`main_tesseract.slurm`: give it folder name and a --lang, such as rus. 

This generates an output file in treain, test, generation folders. separates after a space, maybe a ". " instead of just " "? 






```bash
conda env create python=3.10 -f phi3-environment.yml 
sbatch phi3_model_pipeline.slurm --config_file models/phi3_4k_p1.yaml --fine_tune --prepare_dataset --test
```

## Analyzing Results
Results can be analyzed through the results_analysis.py file, which takes in a list of sheets to be analyzed. For instance, if you have the following files in the `llms_post-ocr_correction-main/results` directory:
* `bart-base.csv`
* `bart-large_50.csv`
* etc.
You would run the following: 
```bash
python results_analysis.py --sheet_names bart-base, bart-large_50
```
NOTE: Only supply the filenames without preceding folder and extension! This is because the code runs `pd.read_csv(f'results/{sheet}.csv')` for each sheet specified in your `sheet_names` list from bash. 


## Other Newton Tips
You can `tmux` into your Newton shell to keep whatever you're doing alive, even across sessions of logging in and out of SSH:
```bash
tmux new -s clocr
```
You can use ctrl-b and then press `[` to be able to scroll with pgUP/pgDown. You can also press ctrl-b and type `:set -g mouse on` if you need to scroll with your mouse. You need to hold down Shift to select text copyable with ctrl+shift+c. ctrl-b+[d] will detach the client  but keep it alive on server. You can come back to it later thorugh `tmux attach clocr`. 