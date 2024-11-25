import pandas as pd

# Formats samples into prompt template using patterns.
# 2D list, with each first-level entry being the fine-tuning prompt followed by the test prompt.
prompts = [
  # Prompt 1 provided by llms_post-ocr_correction GitHub project
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
  # Prompt 2 created by Diego, with modifications by Dr. Bedi
  [ """### Instruction:
The following sentence has grammatical errors, strange words, and spelling mistakes introduced by OCR scanning. Please rewrite it correctly and clearly, without adding new or extraneous tokens.

### Input:
{}

### Response:
{}
""", """### Instruction:
The following sentence has grammatical errors, strange words, and spelling mistakes introduced by OCR scanning. Please rewrite it correctly and clearly, without adding new or extraneous tokens.

### Input:
{}

### Response:
"""],
  # Prompt 3 by Michael Miller
  [ """### Instruction:
The provided text contains mistakes due to OCR processing. Please correct it for readability and accuracy by fixing any errors introduced by OCR, ensuring the result is grammatically correct and clear. Keep the original meaning intact.

### Input:
{}

### Response:
{}
""", """### Instruction:
The provided text contains mistakes due to OCR processing. Please correct it for readability and accuracy by fixing any errors introduced by OCR, ensuring the result is grammatically correct and clear. Keep the original meaning intact.

### Input:
{}

### Response:
"""],
  # Prompt 4 by Michael Miller
  [ """### Instruction:
Edit the following sentence to correct OCR mistakes without altering the total number of characters in the original scanned document. This number is unknown to you, but can be assumed to be the number of characters if everything is spelled correctly and extraneous symbols or corrupted text had its noise rectified.

### Input:
{}

### Response:
{}
""", """### Instruction:
Edit the following sentence to correct OCR mistakes without altering the total number of characters in the original scanned document. This number is unknown to you, but can be assumed to be the number of characters if everything is spelled correctly and extraneous symbols or corrupted text had its noise rectified.

### Input:
{}

### Response:
"""],
  # Prompt 5 by Michael Miller
  [ """### Instruction:
The following input contains mistakes introduced solely by OCR. Please fix any spelling, grammar, or formatting issues in the provided text without adding or removing characters from the original document scanned by OCR.

### Input:
{}

### Response:
{}
""", """### Instruction:
The following input contains mistakes introduced solely by OCR. Please fix any spelling, grammar, or formatting issues in the provided text without adding or removing characters from the original document scanned by OCR.

### Input:
{}

### Response:
"""],
  # Prompt 6 by Michael Miller
  [ """### Instruction:
Revise the following text to correct OCR errors, but maintain the exact character count and avoid introducing any new characters.

### Input:
{}

### Response:
{}
""", """### Instruction:
Revise the following text to correct OCR errors, but maintain the exact character count and avoid introducing any new characters.

### Input:
{}

### Response:
"""],
  # Prompt 7 by Michael Miller
  ["""### Instruction:
Correct only the errors in the provided text. Avoid adding or removing any characters, and keep the character count consistent with what the original document would look like without error noise.

### Input:
{}

### Response:
{}
""", """### Instruction:
Correct only the errors in the provided text. Avoid adding or removing any characters, and keep the character count consistent with what the original document would look like without error noise.

### Input:
{}

### Response:
"""],
  # Prompt 8 takes Prompts 1-7 and runs them through a prompt optimization
  # GPT-4o task, and applies BPO to ensure the prompt is well-engineered
  # from a research-backed perspective. 
  ["""### Instruction:
The following text contains errors introduced by OCR scanning, such as misspellings, grammatical mistakes, and formatting issues. Your task is to correct the text to ensure it is grammatically accurate, readable, and true to its original intended meaning.

**Constraints:**
1. Maintain the exact character count of the corrected text to match what the original, error-free document would contain.
2. Do not introduce any new characters, words, or symbols that were not part of the original input.
3. Correct only the OCR errors without altering the structure or content unnecessarily.

### Input:
{}

### Response:
{}""", """### Instruction:
The following text contains errors introduced by OCR scanning, such as misspellings, grammatical mistakes, and formatting issues. Your task is to correct the text to ensure it is grammatically accurate, readable, and true to its original intended meaning.

**Constraints:**
1. Maintain the exact character count of the corrected text to match what the original, error-free document would contain.
2. Do not introduce any new characters, words, or symbols that were not part of the original input.
3. Correct only the OCR errors without altering the structure or content unnecessarily.

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
