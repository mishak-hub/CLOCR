def format_patterns(model_type, config_path, model_versions, prompt_patterns, datasets_groups, keys_to_print, prompts_to_print):
  for model_version in model_versions:
    for prompt_pattern in prompt_patterns:
      for datagroup in datasets_groups:
        weights_out_path = f"{model_version}_{datagroup}_prompt_{prompt_pattern}"
        statistics_out_path = f"{weights_out_path}.csv"
        filename = f"models/{weights_out_path}.yaml"
        with open(filename, 'w') as file:
          file.write(f"""# Model and Configuration Details
model_type: "{model_type}"               # Specify the model type, e.g., 'BART', 'Llama_2'
config_path: "{config_path}"  # Path to the model's configuration file, local to working dir

# Model Version Details
model_version: "{model_version}" # Model version to load from HuggingFace

# Datasets
datasets:                         # List of datasets with their names
{''.join(f'  - "{dataset}"{chr(10)}' for dataset in datasets_groups[datagroup])}

# Output Paths
weights_out_path: "{weights_out_path}"   # Goes into the `model_weights` folder
statistics_out_path: "{statistics_out_path}" # Goes into the `results` folder

# Prompt Pattern
prompt_pattern: {prompt_pattern}                # Integer representing the selected prompt pattern (see prompts.py)""")
        # print(pattern.format(model_type=model_type, config_path=config_path,
        #             model_version=model_version, datasets=datasets,
        #             weights_out_path=weights_out_path, statistics_out_path=statistics_out_path,
        #             prompt_pattern=prompt_pattern))
        if datagroup in keys_to_print and prompt_pattern in prompts_to_print:
          print(f"sbatch phi3_model_pipeline.slurm --config_file {filename} --fine_tune --test") # don't prepare dataset each time!
        # print("\n\n\n\n---------------------------------\n\n\n")
        
datasets_groups = {"base": ['bln600'], # base dataset style
  "expanded": ['europarl', 'plainwiki', 'plusone', 'iam_tesseract', 'bln600'], # expanded context with 80k-100k entries
  "expanded-10k": ['europarl_10k', 'plainwiki_10k', 'plusone_10k', 'iam_tesseract', 'bln600'], # expanded context but larger datasets reduced by 1/10th size
  "english": ['plainwiki', 'bln600', 'iam_tesseract'],
  "bart-compromise": ['plainwiki_10k', 'bln600'],
  "llama2-compromise": ['europarl_10k', 'plainwiki_10k', 'plusone_10k', 'bln600']
}
config_path = "model_configs.yaml"
prompt_patterns = [1, 8]
keys_to_print = ['expanded-10k'] # this script will make all yaml files, but only print the keys you ask for.
prompts_to_print = [1, 8] # just like keys_to_print
for key in datasets_groups:
    print(f"sbatch phi3_model_pipeline.slurm --config_file models/bart-base_{key}_prompt_1 --prepare_dataset") # prepare dataset once


model_type = "Phi_3"
model_versions = ['phi-3-mini-4k', 'phi-3-mini-128k']
keys_to_print = ['expanded-10k']
prompts_to_print = [8]
format_patterns(model_type, config_path, model_versions, prompt_patterns, datasets_groups, keys_to_print, prompts_to_print)

model_type = "Llama_2"
# model_versions = ['llama-2-7b', 'llama-2-13b', 'llama-2-70b']
model_versions = ['llama-2-7b', 'llama-2-13b']
keys_to_print = ['llama2-compromise']
prompts_to_print = [1, 8] 
format_patterns(model_type, config_path, model_versions, prompt_patterns, datasets_groups, keys_to_print, prompts_to_print)

model_type = "BART"
model_versions = ['bart-base', 'bart-large']
keys_to_print = ['bart-compromise']
prompts_to_print = [1] 
format_patterns(model_type, config_path, model_versions, [1], datasets_groups, keys_to_print, prompts_to_print)

model_type = "Llama_3_1"
# model_versions = ['llama-3.1-7b', 'llama-3.1-13b', 'llama-3.1-70b', 'llama-3.1-405b-instruct']
model_versions = ['llama-3.1-7b', 'llama-3.1-13b']
keys_to_print = ['base']
prompts_to_print = [1]
format_patterns(model_type, config_path, model_versions, prompt_patterns, datasets_groups, keys_to_print, prompts_to_print)

model_versions = ['llama-3.1-13b']
keys_to_print = ['expanded-10k']
prompts_to_print = [8]
format_patterns(model_type, config_path, model_versions, prompt_patterns, datasets_groups, keys_to_print, prompts_to_print)

