# CLOCR

## Prerequisites
* Anaconda: [WSL Installation Guide](https://medium.com/hydroinformatics/software-development-in-linux-install-miniconda-in-wsl-27e809a0c064)

The below sections of installing miniconda are not needed if you just specify a python version! See below section about using `conda env create python=3.10 -f environment.yml `. 

It is quite intimidating to install Anaconda on Newton. I recommend doing the following:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
Now, you should `tmux` into your newton shell to keep whatever you're doing alive:
```bash
tmux new -s jupyter
```
You can use ctrl-b and then press `[` to be able to scroll with pgUP/pgDown. You can also press ctrl-b and type `:set -g mouse on` if you need to scroll with your mouse. You need to hold down Shift to select text copyable with ctrl+shift+c. 

For some reason, Newton doesn't like Conda installing things with multiple threads without being in a slurm job. So, do this:
```bash
srun -p main --time=02:00:00 --ntasks-per-node 2 --gres gpu:1 --pty bash
```
Then, continue through the sections below. If you mess something up and need to restart, and kill the old slurm `bash`, just do this:
```bash
squeue -u cap6614.student2
skill <id_of_job_to_kill>
```

* Rust/Cargo: Just run the following bash command: `curl https://sh.rustup.rs -sSf | sh` ([source](https://doc.rust-lang.org/cargo/getting-started/installation.html))

## To set up llms_post-ocr_correction-main Folder
This folder hosts the training suite. `ls` into that folder, and run:
```bash
conda env create python=3.10 -f environment.yml 
conda activate clocr
```
Newton may give you issues creating the environment. I was able to get it running on my Newton, but I have to check if there were any tweaks I made before running `conda env create`. 

As of now, the code is not finished in implementation, so don't go trying out `pipeline.py` yet!

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