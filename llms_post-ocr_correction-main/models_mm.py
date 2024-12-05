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
# from IPython.core.getipython import get_ipython
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, pipeline
import Levenshtein
# import pandas as pd
# import torch


# phi-3 imports
from transformers import TrainingArguments # AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import ModelCard, ModelCardData, HfApi
from datasets import load_dataset
from jinja2 import Template
# from trl import SFTTrainer
# import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


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

class Phi_3(LanguageModel):
  def __init__(self, config, model=None):
    if model in ['phi-3-mini-4k', 'phi-3-mini-128k']:
      self.model = model
    else:
      print("Phi-3: defaulting to phi-3-mini-4k")
      self.model = 'phi-3-mini-4k'
    self.config = config

  # Load Phi-3 config from YAML file
  def load_config(self, file):
    with open(file, 'r') as f:
      config = yaml.safe_load(f)
    return config['phi-3']

  def fine_tune(self, train_set, weights_out_path, statistics_out_path, prompt_pattern):
    # Load config
    config = self.load_config(self.config)
    finetune_prompt, test_prompt = formatting_func_setup(prompt_pattern)

    # Select model
    model_name = f'microsoft/{self.model.capitalize()}-instruct'
    output_dir = os.path.join('model_weights', weights_out_path)

    if torch.cuda.is_bf16_supported():
      compute_dtype = torch.bfloat16
    else:
      compute_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, force_download=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset = Dataset.from_pandas(train_set)
    print("dataset.info: ", dataset.info)
    print("dataset.features: ", dataset.features)
    print("dataset.column_names: ", dataset.column_names)
    print(f"Number of rows: {len(dataset)}")
    print(f"Number of columns: {len(dataset.column_names)}")

    # This stuff was from the datacamp guide
    """
    dataset = load_dataset(DATASET_NAME, split="train")
    EOS_TOKEN=tokenizer.eos_token_id
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = []
        mapper = {"system": "system\n", "human": "\nuser\n", "gpt": "\nassistant\n"}
        end_mapper = {"system": "", "human": "", "gpt": ""}
        for convo in convos:
            text = "".join(f"{mapper[(turn := x['from'])]} {x['value']}\n{end_mapper[turn]}" for x in convo)
            texts.append(f"{text}{EOS_TOKEN}")
        return {"text": texts}
    dataset = dataset.map(formatting_prompts_func, batched=True)
    print(dataset['text'][8])
    """

    # TODO add bnb and peft+ lora stuff

    config['learning_rate'] = float(config['learning_rate'])
    train_args = TrainingArguments(
      fp16 = not torch.cuda.is_bf16_supported(),
      bf16 = torch.cuda.is_bf16_supported(),
      output_dir=output_dir,
      **config,
    )
    split = dataset.train_test_split(test_size=0.2, seed=42)
    print("split: ", split)
    ds_train = split["train"]
    ds_valid = split["test"]
    print(f"type(dataset): {type(dataset)}\ntype(ds_train): {type(ds_train)}\ntype(ds_valid): {type(ds_valid)}")
    print("ds_train: ", ds_train)
    try:  
      print("ds_train.column_names: ", ds_train.column_names)
      print(f"ds_train Number of rows: {len(ds_train)}")
      print(f"ds_train Number of columns: {len(ds_train.column_names)}")
      print("\n\n")
      print("ds_valid.column_names: ", ds_valid.column_names)
      print(f"ds_valid Number of rows: {len(ds_valid)}")
      print(f"ds_valid Number of columns: {len(ds_valid.column_names)}")
    except AttributeError:
      print("Attribute error for printing information on ds_train, ds_valid")
    try:
      sample = ds_train[0]
      print(f"SAMPLE: {type(sample)}")
      print(sample)
      print("\n\nSAMPLE FINETUNE PROMPT:")
      print(finetune_prompt(sample))
    except:
      print("Couldn't print out a sample for you. Moving on...")


    trainer = SFTTrainer(
      model=model,
      args=train_args,
      train_dataset=ds_train,
      eval_dataset=ds_valid,
      # dataset_text_field="text", # not needed since there's a formatting func taking multiple fields!
      max_seq_length=128,
      formatting_func=finetune_prompt
    )
    # trainer = SFTTrainer(
    #   model=model,
    #   args=train_args,
    #   train_dataset=train,
    #   peft_config=peft_config,
    #   max_seq_length=1024,
    #   tokenizer=tokenizer,
    #   packing=True,
    #   formatting_func=finetune_prompt,
    # )

    trainer.train()
    trainer.save_model(output_dir)

  def test(self, test_set, weights_in_path, statistics_out_path, prompt_pattern):
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
      torch_dtype="auto",
      device_map="cuda",
      force_download=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # i = 0
    # preds = []
    #
    # cell = '''
    # input_ids = tokenizer(test_prompt(test[i]), max_length=1024, return_tensors='pt', truncation=True).input_ids.cuda()
    # with torch.inference_mode():
    #   outputs = model.generate(input_ids=input_ids, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.1, top_k=40)
    # pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):].strip()
    # preds.append(pred)
    # i += 1
    # '''
    #
    # ipython = get_ipython()
    # for _ in tqdm(range(len(test))):
    #     ipython.run_celtl(cell)

    preds = []
    for i in range(len(test)):
        # Prepare input IDs for the model
        prompt = test_prompt(test[i])
        input_ids = tokenizer(prompt, max_length=128, return_tensors='pt', truncation=True).input_ids.cuda()
        # Run inference
        with torch.inference_mode():
            outputs = model.generae(
                input_ids=input_ids,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.1,
                top_k=40
            )

        # Decode output and store the result
        pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):].strip()
        preds.append(pred)

    results = self.get_results(test, preds)
    results.to_csv(f'results/{statistics_out_path}', index=False)


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
    print("train.info: ", train.info)
    print("train.features: ", train.features)
    print("train.column_names: ", train.column_names)
    print(f"Number of rows: {len(train)}")
    print(f"Number of columns: {len(train.column_names)}")

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
      force_download=True
    )
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    # added at chatgpt's recommendation:
    max_seq_length = 512
    tokenizer.model_max_length = max_seq_length
    tokenizer.truncation = False

    sample = train[0]
    print(f"SAMPLE: {type(sample)}")
    print(sample)
    print("SAMPLE FINETUNE PROMPT:")
    print(finetune_prompt(sample))


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
      max_seq_length=max_seq_length,
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

    #i = 0
    #preds = []

    #cell = '''
    #input_ids = tokenizer(test_prompt(test[i]), max_length=1024, return_tensors='pt', truncation=True).input_ids.cuda()
    #with torch.inference_mode():
    #  outputs = model.generate(input_ids=input_ids, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.1, top_k=40)
    #pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):].strip()
    #preds.append(pred)
    #i += 1
    #'''

    #ipython = get_ipython()
    #for _ in tqdm(range(len(test))):
    #    ipython.run_cell(cell)
    preds = []
    for i in range(len(test)):
        # Prepare input IDs for the model
        prompt = test_prompt(test[i])
        input_ids = tokenizer(prompt, max_length=1024, return_tensors='pt', truncation=True).input_ids.cuda()
        # Run inference
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.1,
                top_k=40
            )

        # Decode output and store the result
        pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):].strip()
        preds.append(pred)

    results = self.get_results(test, preds)
    results.to_csv(f'results/{statistics_out_path}', index=False)


#AK
class Llama_3(LanguageModel):
  def __init__(self, config, model=None):
    if model in ['llama-3-8b', 'llama-3-8b-instruct', 'llama-3-70b', 'llama-3-70b-instruct', 'llama-3.1-8b', 'llama-3-8b-instruct', 'llama-3.1-70b', 'llama-3.1-70b-instruct', 'llama-3.1-405b', 'llama-3.1-405b-instruct', 'llama-3.2-1b', 'llama-3.2-1b-instruct', 'llama-3.2-3b', 'llama-3.2-3b-instruct']:
      self.model = model
    else:
      print("Llama-3: defaulting to llama-3-8b")
      self.model = 'llama-3-8b'
    self.config = config

  # Load Llama 3 config from YAML file
  def load_config(self, file):
    with open(file, 'r') as f:
      config = yaml.safe_load(f)
    return config['llama-3']

  def fine_tune(self, train_set, weights_out_path, statistics_out_path, prompt_pattern):
    # Load config
    config = self.load_config(self.config)
    finetune_prompt, test_prompt = formatting_func_setup(prompt_pattern)

    # Select model
    if self.model in ['llama-3-8b', 'llama-3-8b-instruct', 'llama-3-70b', 'llama-3-70b-instruct']:
      model_name = f'meta-llama/Meta-{self.model.title()}'
    else:
      model_name = f'meta-llama/{self.model.title()}'
    #model_name = f'meta-llama/Meta-{self.model.capitalize()}'
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

    # Initialise Llama 3
    model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=bnb_config,
      use_cache=False,
      device_map='auto',
      force_download=True
    )
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Instruction-tune Llama 3
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
      force_download=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    #i = 0
    #preds = []

    #cell = '''
    #input_ids = tokenizer(test_prompt(test[i]), max_length=1024, return_tensors='pt', truncation=True).input_ids.cuda()
    #with torch.inference_mode():
    #  outputs = model.generate(input_ids=input_ids, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.1, top_k=40)
    #pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):].strip()
    #preds.append(pred)
    #i += 1
    #'''

    #ipython = get_ipython()
    #for _ in tqdm(range(len(test))):
    #    ipython.run_cell(cell)

    preds = []
    for i in range(len(test)):
        # Prepare input IDs for the model
        prompt = test_prompt(test[i])
        input_ids = tokenizer(prompt, max_length=1024, return_tensors='pt', truncation=True).input_ids.cuda()
        # Run inference
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.1,
                top_k=40
            )

        # Decode output and store the result
        pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):].strip()
        preds.append(pred)
        

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
      generator = pipeline('text2text-generation', model=model.to('cuda'), tokenizer=tokenizer, device='cuda', max_length=1024, batch_size=32) # TODO This may be broken

      preds = []
      for sample in tqdm(test):
        preds.append(generator(sample['OCR Text'], batch_size=32)[0]['generated_text'])

      results = self.get_results(test, preds)
      results.to_csv(f'results/{statistics_out_path}', index=False)

# MM; this class is unused. Replaced with Aleenah's Llama_3 class. 
class Llama_3_1(LanguageModel):
  def __init__(self, config, model=None):
    # Check if the specified model is available; default to 'llama-3.1-405b-instruct'
    if model in ['llama-3.1-7b', 'llama-3.1-13b', 'llama-3.1-70b', 'llama-3.1-405b-instruct']:
      self.model = model
    else:
      print("Llama-3.1: defaulting to llama-3.1-405b-instruct")
      self.model = 'llama-3.1-405b-instruct'
    self.config = config

  # Load Llama 3.1 config from YAML file
  def load_config(self, file):
    with open(file, 'r') as f:
      config = yaml.safe_load(f)
    return config['llama-3.1']

  def fine_tune(self, train_set, weights_out_path, statistics_out_path, prompt_pattern):
    # Load config
    config = self.load_config(self.config)
    finetune_prompt, test_prompt = formatting_func_setup(prompt_pattern)

    # Select model name for Hugging Face
    model_name = f'meta-llama/{self.model.capitalize()}-hf'
    output_dir = os.path.join('model_weights', weights_out_path)

    # Set up training data
    train = Dataset.from_pandas(train_set)

    # Quantization config for large models
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type='nf4',
      bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # LoRA configuration
    peft_config = LoraConfig(
      lora_alpha=32,            # Increased alpha for larger model
      lora_dropout=0.15,        # Higher dropout to avoid overfitting
      r=128,                    # Higher r for larger layer adjustments
      bias='none',
      task_type='CAUSAL_LM',
    )

    # Initialize Llama 3.1
    model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config=bnb_config,
      use_cache=False,
      device_map='auto',
      force_download=True
    )
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Instruction-tune Llama 3.1
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
    # Model and paths setup
    model_dir = os.path.join('model_weights', weights_in_path)
    finetune_prompt, test_prompt = formatting_func_setup(prompt_pattern)

    # Load test dataset
    test = Dataset.from_pandas(test_set)

    # Quantization config
    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type='nf4',
      bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load fine-tuned model
    model = AutoPeftModelForCausalLM.from_pretrained(
      model_dir,
      quantization_config=bnb_config,
      low_cpu_mem_usage=True,
      device_map='auto',
      torch_dtype=torch.bfloat16,
      force_download=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Collect predictions
    preds = []
    for i in tqdm(range(len(test))):
      input_text = test_prompt(test[i])  # Format input for the current test instance
      input_ids = tokenizer(input_text, max_length=1024, return_tensors='pt', truncation=True).input_ids.cuda()

      # Generate predictions
      with torch.inference_mode():
        outputs = model.generate(
          input_ids=input_ids,
          max_new_tokens=1024,
          do_sample=True,
          temperature=0.7,
          top_p=0.1,
          top_k=40
        )
      # Decode prediction
      pred = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(input_text):].strip()
      preds.append(pred)

    # Evaluate results
    results = self.get_results(test, preds)
    results.to_csv(f'results/{statistics_out_path}', index=False)