import pandas as pd

# Formats samples into prompt template using patterns.
# 2D list, with each first-level entry being the fine-tuning prompt followed by the test prompt.
prompts = [
  # Prompt provided by llms_post-ocr_correction GitHub project
  [ """### Instruction:
Fix the OCR errors in the provided text.

### Input:
{}

### Response:
{}
""", """### Instruction:
Fix the OCR errors in the provided text.

### Input:
{}

### Response:
"""], 
  # Prompt created by Diego, with modifications by Dr. Bedi
  [ """### Instruction:
The following sentence has grammatical errors, strange words, and spelling mistakes. Please rewrite it correctly and clearly, without adding new or extraneous tokens.

### Input:
{}

### Response:
{}
""", """### Instruction:
The following sentence has grammatical errors, strange words, and spelling mistakes. Please rewrite it correctly and clearly, without adding new or extraneous tokens.

### Input:
{}

### Response:
"""]
]

# These two functions are for models that don't require SFTTrainer
def finetune_prompt(prompt: int, example: pd.DataFrame):
  return prompts[prompt][0].format(example['OCR Text'], example['Ground Truth'])

def test_prompt(prompt: int, test_sample: pd.DataFrame):
  return prompts[prompt][1].format(test_sample['OCR Text'])


# This function is for generating the above functions with only one variable
# per finetune_prompt call. For use in things like SFTTrainer, which doesn't
# allow args to be used. 
# usage: finetune_prompt, test_prompt = formatting_func_setup(prompt);
#        formatting_func=finetune_prompt; # in SFTTrainer
def formatting_func_setup(prompt: int):
  def ft_prompt(example: pd.DataFrame):
    return prompts[prompt][0].format(example['OCR Text'], example['Ground Truth'])
  def t_prompt(test_sample: pd.DataFrame):
    return prompts[prompt][1].format(test_sample['OCR Text'])
  
  return ft_prompt, t_prompt