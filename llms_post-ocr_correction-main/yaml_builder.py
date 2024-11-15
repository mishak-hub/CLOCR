def format_patterns(model_type, config_path, model_versions, prompt_patterns, datasets_groups):
  for model_version in model_versions:
    for prompt_pattern in prompt_patterns:
      for i, datasets in enumerate(datasets_groups):
        weights_out_path = f"{model_version}_datagroup_{i}_prompt_{prompt_pattern}"
        statistics_out_path = f"{weights_out_path}.csv"
        filename = f"final_models/{weights_out_path}.yaml"
        with open(filename, 'w') as file:
          file.write(f"""# Model and Configuration Details
model_type: "{model_type}"               # Specify the model type, e.g., 'BART', 'Llama_2'
config_path: "{config_path}"  # Path to the model's configuration file, local to working dir

# Model Version Details
model_version: "{model_version}" # Model version to load from HuggingFace

# Datasets
datasets:                         # List of datasets with their names
{''.join(f'  - "{dataset}"{chr(10)}' for dataset in datasets)}

# Output Paths
weights_out_path: "{weights_out_path}"   # Goes into the `model_weights` folder
statistics_out_path: "{statistics_out_path}" # Goes into the `results` folder

# Prompt Pattern
prompt_pattern: {prompt_pattern}                # Integer representing the selected prompt pattern (see prompts.py)""")
        # print(pattern.format(model_type=model_type, config_path=config_path,
        #             model_version=model_version, datasets=datasets,
        #             weights_out_path=weights_out_path, statistics_out_path=statistics_out_path,
        #             prompt_pattern=prompt_pattern))
        print(f"sbatch phi3_model_pipeline.slurm --config_file final_models/{filename} --fine_tune --test") # don't prepare dataset each time!
        # print("\n\n\n\n---------------------------------\n\n\n")

datasets_groups = [['iam_tesseract', 'bln600'],
  ['europarl_10k', 'plainwiki_10k', 'plusone_10k', 'iam_tesseract', 'bln600'],
  ['europarl', 'plainwiki', 'plusone', 'iam_tesseract', 'bln600']]
config_path = "model_configs.yaml"
prompt_patterns = range(1, 4)
for i in range(len(datasets_groups)):
    print(f"sbatch phi3_model_pipeline.slurm --config_file models/bart-base_datagroup_{i}_prompt_1 --prepare_dataset") # prepare dataset once


model_type = "Phi_3"
model_versions = ['phi-3-mini-4k', 'phi-3-mini-128k']
format_patterns(model_type, config_path, model_versions, prompt_patterns, datasets_groups)

model_type = "Llama_2"
model_versions = ['llama-2-7b', 'llama-2-13b', 'llama-2-70b']
format_patterns(model_type, config_path, model_versions, prompt_patterns, datasets_groups)

model_type = "BART"
model_versions = ['bart-base', 'bart-large']
format_patterns(model_type, config_path, model_versions, prompt_patterns, datasets_groups)

model_type = "Llama_3_1"
model_versions = ['llama-3.1-7b', 'llama-3.1-13b', 'llama-3.1-70b', 'llama-3.1-405b-instruct']
format_patterns(model_type, config_path, model_versions, prompt_patterns, datasets_groups)
