from abc import ABC, abstractmethod
from prompts import formatting_func_setup
# llama-2 imports
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import argparse
import os
import pandas as pd
import torch
import yaml

# bart imports
# from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
# import argparse
# import os
# import pandas as pd
# import yaml

# results notebook imports
# from datasets import Dataset
from IPython.core.getipython import get_ipython
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, pipeline
import Levenshtein
# import pandas as pd
# import torch

class LanguageModel(ABC):

  # @abstractmethod
  # def __init__(self):
  #     pass
  @abstractmethod
  def fine_tune(self, train_set, weights_out_path, statistics_out_path, prompt_pattern):
      pass
  @abstractmethod
  def test(self, test_set, weights_in_path, statistics_out_path, prompt_pattern):
      pass
  # @abstractmethod
  # def validate(self, val_set, weights_in_path, statistics_out_path, prompt_pattern):
  #     pass
    
  # Compute character error rate (CER)
  def cer(self, prediction, target):
    distance = Levenshtein.distance(prediction, target)
    return distance / len(target)

  # Helper function to store results as a CSV
  def get_results(self, data, preds):
    results = data.to_pandas()
    results['Model Correction'] = preds
    results = results.rename(columns={'CER': 'old_CER'})
    results['new_CER'] = results.apply(lambda row: self.cer(row['Model Correction'], row['Ground Truth']), axis=1)
    results['CER_reduction'] = ((results['old_CER'] - results['new_CER']) / results['old_CER']) * 100
    return results

# TODO: args.model -> self.model may conflict with test function's use of local model variable

class Llama_2(LanguageModel):
  def __init__(self, config, model=None):
    if model in ['llama-2-7b', 'llama-2-13b', 'llama-2-70b']:
      self.model = model
    else:
      print("Llama-2: defaulting to llama-2-7b")
      self.model = 'llama-2-7b'
    self.config = config
    
  # Load Llama 2 config from YAML file
  def load_config(self, file):
    with open(file, 'r') as f:
      config = yaml.safe_load(f)
    return config['llama-2']
  
  def fine_tune(self, train_set, weights_out_path, statistics_out_path, prompt_pattern):
    # Load config
    config = self.load_config(self.config)
    finetune_prompt, test_prompt = formatting_func_setup(prompt_pattern)

    # Select model
    model_name = f'meta-llama/{self.model.capitalize()}-hf'
    output_dir = os.path.join('model_weights', weights_out_path)

    # Set up training data
    train = Dataset.from_pandas(train_set)

    # Quantization config
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type='nf4',
      bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # LoRA config
    peft_config = LoraConfig(
      lora_alpha=16,
      lora_dropout=0.1,
      r=64,
      bias='none',
      task_type='CAUSAL_LM',
    )

    # Initialise Llama 2
    model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=bnb_config,
      use_cache=False,
      device_map='auto',
    )
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Instruction-tune Llama 2
    config['learning_rate'] = float(config['learning_rate'])
    train_args = SFTConfig(
      output_dir=output_dir,
      **config,
    )
    trainer = SFTTrainer(
      model=model,
      args=train_args,
      train_dataset=train,
      peft_config=peft_config,
      max_seq_length=1024,
      tokenizer=tokenizer,
      packing=True,
      formatting_func=finetune_prompt,
    )
    trainer.train()
    trainer.save_model(output_dir)
    
  def test(self, test_set, weights_in_path, statistics_out_path, prompt_pattern):
    # model_dir = 'pykale/llama-2-13b-ocr' # their pretrained model
    model_dir = os.path.join('model_weights', weights_in_path)
    finetune_prompt, test_prompt = formatting_func_setup(prompt_pattern)
    
    test = Dataset.from_pandas(test_set)

    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type='nf4',
      bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
      model_dir,
      quantization_config=bnb_config,
      low_cpu_mem_usage=True,
      torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    i = 0
    preds = []

    cell = '''
    input_ids = tokenizer(test_prompt(test[i]), max_length=1024, return_tensors='pt', truncation=True).input_ids.cuda()
    with torch.inference_mode():
      outputs = model.generate(input_ids=input_ids, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.1, top_k=40)
    pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):].strip()
    preds.append(pred)
    i += 1
    '''

    ipython = get_ipython()
    for _ in tqdm(range(len(test))):
        ipython.run_cell(cell)

    results = self.get_results(test, preds)
    results.to_csv(f'results/{statistics_out_path}', index=False)
    
    
class BART(LanguageModel):
  def __init__(self, config, model=None):
    if model in ['bart-base', 'bart-large']:
      self.model = model
    else:
      print("BART: defaulting to bart-base")
      self.model = 'bart-base'
    self.config = config
  
  # Load BART config from YAML file
  def load_config(self, file):
    with open(file, 'r') as f:
      config = yaml.safe_load(f)
    return config['bart']
    
  def fine_tune(self, train_set, weights_out_path, statistics_out_path, prompt_pattern):
    # Load config
    config = self.load_config(self.config)

    # Select model
    model_name = f'facebook/{self.model}'
    output_dir = os.path.join('model_weights', weights_out_path)

    # Set up training data
    train = train_set
    train['text'] = train['OCR Text']
    train['labels'] = train['Ground Truth']
    train = Dataset.from_pandas(train)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train = train.map(lambda x: tokenizer(x['text'], text_target=x['labels'], max_length=1024, truncation=True), batched=True)

    # Initialise BART
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Fine-tune BART
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    config['learning_rate'] = float(config['learning_rate'])
    train_args = Seq2SeqTrainingArguments(
      output_dir=output_dir,
      **config,
    )
    trainer = Seq2SeqTrainer(
      model,
      train_args,
      train_dataset=train,
      data_collator=data_collator,
      tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    
  def test(self, test_set, weights_in_path, statistics_out_path, prompt_pattern):
      # model_dir = 'pykale/bart-large-ocr' # their pre-trained model
      model_dir = os.path.join('model_weights', weights_in_path)

      test = test_set
      test = Dataset.from_pandas(test)

      model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
      tokenizer = AutoTokenizer.from_pretrained(model_dir)
      generator = pipeline('text2text-generation', model=model.to('cuda'), tokenizer=tokenizer, device='cuda', max_length=1024, batch_size=8) # TODO This may be broken

      preds = []
      for sample in tqdm(test):
        preds.append(generator(sample['OCR Text'])[0]['generated_text'])

      results = self.get_results(test, preds)
      results.to_csv(f'results/{statistics_out_path}', index=False)