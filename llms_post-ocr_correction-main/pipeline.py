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

from prompts import finetune_prompt, test_prompt
from models import LanguageModel, Llama_2, BART

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

def split_dataset(seq: pd.DataFrame, name: str):
  """Splits dataset `seq` into train/test/val split of 80%, 17.5%, 2.5%, saving into data/`name` folder path. 

  Args:
      seq (pd.DataFrame): OCR-GT pairs in a columnar format. 
      name (str): The name you want to save this dataset as. 
  """
  seq['OCR Text'] = preprocess(seq['OCR Text'])
  seq['Ground Truth'] = preprocess(seq['Ground Truth'])
  train_ids, test_ids = train_test_split(seq['Sample ID'].unique(), test_size=0.2, random_state=600)
  train_ids, val_ids = train_test_split(train_ids, test_size=0.125, random_state=600)

  train = seq[seq['Sample ID'].isin(train_ids)]
  test = seq[seq['Sample ID'].isin(test_ids)]
  val = seq[seq['Sample ID'].isin(val_ids)]
  train.to_csv(f'data/{name}_train.csv', index=False)
  test.to_csv(f'data/{name}_test.csv', index=False)
  val.to_csv(f'data/{name}_val.csv', index=False)


def suite(datasets: list[str], language_model: LanguageModel, weights_out_path: os.PathLike, statistics_out_path: os.PathLike, prompt_pattern: int):
  # split up each dataset into train/test/val
  train_set = [] # list of pd tables
  test_set = [] # list of pd tables
  val_set = [] # list of pd tables
  for dataset in datasets:
    seq = pd.read_csv(f'datasets/{dataset}.csv') # use pandas probably
    train, test, val = split_dataset(seq)
    train_set.append(train)
    test_set.append(test)
    val_set.append(val)
  
  # Launch fine-tune script relevant to LM
  language_model.fine_tune(train_set, test_set, weights_out_path, statistics_out_path, prompt_pattern)  
  
  # Test performance of LM
  language_model.test(test_set, weights_in_path, statistics_out_path, prompt_pattern)
    
  # Perform validation (currently unused)
  # language_model.validate(val_set, weights_in_path, statistics_out_path, prompt_pattern)  
  
def prepare_suite(args):
  # args: 
  #     * model_type: what LM you are loading. Llama_2 and BART atm.
  #     * config: Config.yaml path
  #     * model_version: model version to load from HuggingFace
  #     * datasets: A list of dataset names with a parent corpus CSV in datasets folder. 
  #     * weights_out_path: Where model weights will be stored. L2 and BART disregard this.
  #     * statistics_out_path: Where model statistics will be stored. L2 and BART disregard this.
  #     * prompt_pattern: an integer selecting enumerated prompt pattern found in prompts.py.
  
  # load up LM (if/elif tree)
  if (args.model_type == "Llama_2"):
    language_model = Llama_2(args.config, args.model_version)
  elif (args.model_type == "BART"):
    language_model = BART(args.config, args.model_version)
  else:
    raise ValueError("Only 'Llama_2' and 'BART' models supported currently!")
  
  suite(args.datasets, language_model, args.weights_out_path, args.statistics_out_path, args.prompt_pattern)

  
  
def observe_ocr_paper_correction_results():
  """This function just shows the results for the four models trained and tested by the llms_post-ocr_correction project's team. Taken from their results.ipynb file.
  """
  results = {'bart-base': pd.read_csv('results/bart-base.csv'),
            'bart-large': pd.read_csv('results/bart-large.csv'),
            'llama-2-7b': pd.read_csv('results/llama-2-7b.csv'),
            'llama-2-13b': pd.read_csv('results/llama-2-13b.csv')}

  corrections = results['llama-2-13b']
  corrections.head(10)
  
  for i in range(len(corrections)):
    print(i+1)
    print(f"OCR Text:\n{corrections['OCR Text'][i]}\n")
    print(f"Ground Truth:\n{corrections['Ground Truth'][i]}\n")
    print(f"Model Correction:\n{corrections['Model Correction'][i]}\n\n")
    
if __name__ == '__main__':
  # Parse arguments for model/config/data
  parser = argparse.ArgumentParser(description='Fine-tuning and training an LM')
  # parser.add_argument('--model', type=str, choices=['bart-base', 'bart-large'],
  #                     default='bart-base', help='Specify model: bart-base, bart-large')
  # parser.add_argument('--config', type=str, help='Path to config')
  # parser.add_argument('--data', type=str, help='Path to training data')
  parser.add_argument('--model_type', type=str, help='Specify model: BART, Llama_2', 
                      choices=['Llama_2', 'BART'], )
  
  parser.add_argument('--config', type=str, help='Path to config.yaml')
  parser.add_argument('--model_version', type=str, help='Model version to load from HuggingFace')
  parser.add_argument('--datasets', type=list[str], help='A list of dataset names with a parent corpus CSV in datasets folder.')
  parser.add_argument('--weights_out_path', type=str, help='Where model weights will be stored. L2 and BART disregard this.')
  parser.add_argument('--statistics_out_path', type=str, help='Where model statistics will be stored. L2 and BART disregard this.')
  parser.add_argument('--prompt_pattern', type=str, help='An integer selecting enumerated prompt pattern found in prompts.py.')
  args = parser.parse_args()

  prepare_suite(args)
  
  # python pipeline.py --model_type "BART" --config config.yaml --model_version "bart-base" --datasets "bln600" --weights_out_path "no" --statistics_out_path "no" --prompt_pattern 1