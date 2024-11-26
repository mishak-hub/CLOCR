"""
What are our goals? We want to test out different combinations of:
* datasets
* Language Models
* LM prompts
and chiefly, compare the work of the models to each other as a
benchmark for performance. This is "better" for most commercial
use cases than training an entire VLM on a bunch of video
data + textual ground truth, because it uses free/cheap LMs and 
OCR models which already exist. 
  We will do some side analysis on different OCR models, but it is
not the goal of this project. There is benefit to training on the 
output issues of *many* OCR models because it allows the LM to 
better-understand the broad error patterns involved in OCR in general.
However, there may also be benefit to fine-tuning an LM only on 
output from the OCR model being used in a real product, because
competing OCR algorithms may have different patterns of error which
are not present in the one being used. Learning those patterns may
mislead the LM during fine-tuning to see certain things as errors when
they are not. This may or may not be true, so we could also benchmark
that phenomemon.

------------

Pipeline parts:
1. Create dataset: These are (OCR output, ground truth) pairs. 
  Parameters: corpus, num_samples, ocr_model, output_path
  For some input corpus:
    * Generate OCR input images from many text samples, recording GT
    * Perform OCR using model specified
    * Export the results as a list of (OCR output, ground truth) pairs. Likely a CSV.
  
  This is done in other files. TBD. 
  
2. Fine-tune LM on train-test split: Teach the LM how to fix OCR issues.
  Parameters: datasets, datasets_split_out_path, language_model, weights_out_path, statistics_out_path, prompt_pattern
  Algorithm:
    * Create train-test-val split and save setup for later validation phase
    * Use the LM API to fine-tune on the datasets using the prompt pattern given.
    * Export the new model weights or whatever is needed to reuse this model.
    
  This is performed below.
  
3. Validate LM performance on val section from train-test_split:
  Parameters: (?)datasets, datasets_split_in_path, language_model, weights_in_path, prompt_pattern, statistics_out_path
  Algorithm:
    * Load the model weights and validation dataset split
    * use the LM selected to record validation performance
    * Export the statistics
    
  This is performed below.

------------

  We can fine-tune on either one prompt or a random mix of all prompts
to see if any are better than the others. Mixing probably wouldn't work
because we will only use one prompt in the validation segment. 
  Additionally, we'll see if any sort of dataset produces better results.
"""
from sklearn.model_selection import train_test_split
import json
import Levenshtein
import os
import pandas as pd
import argparse
import yaml
from typing import Tuple

from prompts import finetune_prompt, test_prompt
from models import LanguageModel, Llama_2, BART, Phi_3, Llama_3

def load_config(config_path: os.PathLike):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# Compute character error rate (CER)
def cer(prediction, target):
    distance = Levenshtein.distance(prediction, target)
    return distance / len(target)

# Helper function to preprocess text
def preprocess(c):
    c = c.str.replace("‘", "'", regex=False)
    c = c.str.replace("’", "'", regex=False)
    c = c.str.replace("“", '"', regex=False)
    c = c.str.replace("”", '"', regex=False)
    c = c.str.replace("—", "-", regex=False)
    c = c.str.replace(r'\s+', ' ', regex=True)
    c = c.str.strip()
    return c

def split_dataset(seq: pd.DataFrame, name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Splits dataset `seq` into train/test split of 80%, 20%, saving into data/`name` folder path. 

  Args:
      seq (pd.DataFrame): OCR-GT pairs in a columnar format. 
      name (str): The name you want to save this dataset as. 
      
  Returns:
      Tuple[pd.DataFrame, pd.DataFrame]: (training, testing) set from split.
  """
  seq['OCR Text'] = preprocess(seq['OCR Text'])
  seq['Ground Truth'] = preprocess(seq['Ground Truth'])
  train_ids, test_ids = train_test_split(seq['Sample ID'].unique(), test_size=0.2, random_state=600)
  # train_ids, val_ids = train_test_split(train_ids, test_size=0.125, random_state=600) # formerly used for 80%:17.5%:2.5% train/test/val split

  train = seq[seq['Sample ID'].isin(train_ids)]
  test = seq[seq['Sample ID'].isin(test_ids)]
  # val = seq[seq['Sample ID'].isin(val_ids)]
  
  train['CER'] = train.apply(lambda row: cer(row['OCR Text'], row['Ground Truth']), axis=1)
  test['CER'] = test.apply(lambda row: cer(row['OCR Text'], row['Ground Truth']), axis=1)
  
  train.to_csv(f'data/{name}_train.csv', index=False)
  test.to_csv(f'data/{name}_test.csv', index=False)
  # val.to_csv(f'data/{name}_val.csv', index=False)
  return train, test
  
def prepare_datasets(model_type: str, config_path: os.PathLike | str, model_version: str, 
            datasets: list[str], weights_out_path: os.PathLike | str, 
            statistics_out_path: os.PathLike | str, prompt_pattern: int):
  # split up each dataset into train/test
  train_set = pd.DataFrame(columns=['OCR Text','Ground Truth','Sample ID'])
  test_set = pd.DataFrame(columns=['OCR Text','Ground Truth','Sample ID'])
  for dataset in datasets:
    seq = pd.read_csv(f'datasets/{dataset}.csv') # use pandas probably
    train, test = split_dataset(seq, dataset)
    train_set = train_set._append(train, ignore_index=True)
    test_set = test_set._append(test, ignore_index=True)
    # val_set._append(val)
  
  train_set['CER'] = train_set.apply(lambda row: cer(row['OCR Text'], row['Ground Truth']), axis=1)
  test_set['CER'] = test_set.apply(lambda row: cer(row['OCR Text'], row['Ground Truth']), axis=1)

  print(f"""Prepared {len(datasets)} datasets for model. Total sizes:
        train: {train_set.count}; CER stats: {train_set['CER'].describe()}
        test: {test_set.count}; CER stats: {test_set['CER'].describe()}""")
  
  
def load_datasets(datasets: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Takes a list of dataset names (NOT PATHS) and loads them as DataFrames.

  Args:
      datasets (list[str]): Dataset names that have already been processed by `prepare_datasets` task, meaning a train and test split is present.

  Returns:
      Tuple[pd.DataFrame, pd.DataFrame]: train, test dfs.
  """
  train_set = pd.DataFrame(columns=['OCR Text','Ground Truth','Sample ID'])
  test_set = pd.DataFrame(columns=['OCR Text','Ground Truth','Sample ID'])
  for dataset in datasets:
    train = pd.read_csv(f'data/{dataset}_train.csv', usecols=['CER','OCR Text','Ground Truth','Sample ID'])
    test = pd.read_csv(f'data/{dataset}_test.csv', usecols=['CER','OCR Text','Ground Truth','Sample ID'])
    train['Sample ID'] = train['Sample ID'].astype(str)
    test['Sample ID'] = test['Sample ID'].astype(str)
    train_set = train_set._append(train, ignore_index=True)
    test_set = test_set._append(test, ignore_index=True)
    
  print("train_set:")
  print(train_set.head())
  print(train_set.tail())
  print("test_set:")
  print(test_set.head())
  print(test_set.tail())
  
  return train_set, test_set
  
  
def load_model(model_type: str, config_path: os.PathLike | str, 
               model_version: str) -> LanguageModel:
  """Loads LM for train/test tasks.

  Args:
      model_type (str): The model you want to load, such as Llama_2 or BART.
      config_path (os.PathLike | str): Path to model_configs.yaml file. 
      model_version (str): Specific model version, such as bart-large. 

  Raises:
      ValueError: If you do not supply a legal model_type.

  Returns:
      LanguageModel: The LM loaded upon successful init.
  """
  if (model_type == "Llama_2"):
    language_model = Llama_2(config_path, model_version)
  elif (model_type == "Phi_3"):
    language_model = Phi_3(config_path, model_version)
  elif (model_type == "Llama_3_1"):
    language_model = Llama_3(config_path, model_version)
  elif (model_type == "BART"):
    language_model = BART(config_path, model_version)
  else:
      raise ValueError("Only 'Phi_3', 'Llama_2', 'Llama_3_1' and 'BART' models supported currently!")  
  
  return language_model
  

def fine_tune(model_type: str, config_path: os.PathLike | str, model_version: str, 
            datasets: list[str], weights_out_path: os.PathLike | str, 
            statistics_out_path: os.PathLike | str, prompt_pattern: int):
  """Runs fine-tuning task on some list of datasets, prompt, and LM model. 

  Args:
      model_type (str): What LM you are loading; Llama_2 and BART atm.
      config_path (os.PathLike | str): Path to model_configs.yaml file. 
      model_version (str): Model version to load from HuggingFace, such as bart-large.
      datasets (list[str]): A list of dataset names which have been processed from datasets/ to data/ folder. 
      weights_out_path (os.PathLike | str): Where model weights will be stored; optional and unused by some models.
      statistics_out_path (os.PathLike | str): Where model statistics will be stored; optional and unused by some models.
      prompt_pattern (int): An integer selecting enumerated prompt pattern found in prompts.py.
  """  
  # load up datasets
  train_set, test_set = load_datasets(datasets)
  
  # load up model
  language_model = load_model(model_type, config_path, model_version)
  
  # Launch fine-tune script relevant to LM
  language_model.fine_tune(train_set, weights_out_path, statistics_out_path, prompt_pattern) 


def test(model_type: str, config_path: os.PathLike | str, model_version: str, 
            datasets: list[str], weights_in_path: os.PathLike | str, 
            statistics_out_path: os.PathLike | str, prompt_pattern: int):
  # load up datasets
  train_set, test_set = load_datasets(datasets)
  
  # load up model
  language_model = load_model(model_type, config_path, model_version)
  
  # Test performance of LM
  language_model.test(test_set, weights_in_path, statistics_out_path, prompt_pattern)

    
if __name__ == '__main__':
  # Parse arguments for model/config/data
  parser = argparse.ArgumentParser(description='Fine-tuning and training an LM')
  parser.add_argument("task", type=str, choices=['prepare_datasets', 'fine_tune', 'test'], 
                      help='Specify task for model to perform: prepare_datasets, fine_tune, or test.')
  parser.add_argument("config_file", type=str, 
                      help="Path to the test YAML configuration file.")
  args = parser.parse_args()
  
  config = load_config(args.config_file)
    
  # Extract parameters from the config file
  model_type = config.get("model_type")
  config_path = config.get("config_path")
  model_version = config.get("model_version")
  datasets = config.get("datasets")# , []
  weights_out_path = config.get("weights_out_path", None)
  statistics_out_path = config.get("statistics_out_path", None)
  prompt_pattern = config.get("prompt_pattern")
  
  # print(config)

  # Print the loaded configuration (for debugging purposes)
  print(f"Model Type: {model_type}")
  print(f"Config Path: {config_path}")
  print(f"Model Version: {model_version}")
  print(f"Datasets: {datasets}")
  print(f"Weights Output Path: {weights_out_path}")
  print(f"Statistics Output Path: {statistics_out_path}")
  print(f"Prompt Pattern: {prompt_pattern}")
  
  # Validate required parameters
  if (not model_type or not config_path or not model_version 
    or not datasets or prompt_pattern is None):
    raise ValueError("Missing required parameters in the configuration yaml file.")
  
  if (args.task == "prepare_datasets"):
    prepare_datasets(model_type, config_path, model_version, datasets, weights_out_path, statistics_out_path, prompt_pattern)
  elif (args.task == "fine_tune"):
    fine_tune(model_type, config_path, model_version, datasets, weights_out_path, statistics_out_path, prompt_pattern)
  elif (args.task == "test"):
    test(model_type, config_path, model_version, datasets, weights_out_path, statistics_out_path, prompt_pattern)
  
  # python pipeline.py --model_type "BART" --config config.yaml --model_version "bart-base" --datasets "bln600" --weights_out_path "no" --statistics_out_path "no" --prompt_pattern 1
  
  # python pipeline.py prepare_datasets models/bart-base.yaml
