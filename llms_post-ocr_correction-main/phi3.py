from models import LanguageModel



from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from huggingface_hub import ModelCard, ModelCardData, HfApi
from datasets import load_dataset
from jinja2 import Template
from trl import SFTTrainer
import yaml
import torch
     
def fine_tune():
  MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
  NEW_MODEL_NAME = "opus-samantha-phi-3-mini-4k"
      

  DATASET_NAME = "macadeliccc/opus_samantha"
  SPLIT = "train"
  MAX_SEQ_LENGTH = 2048
  num_train_epochs = 1
  license = "apache-2.0"
  username = "zoumana"
  learning_rate = 1.41e-5
  per_device_train_batch_size = 4
  gradient_accumulation_steps = 1
      

  if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
  else:
    compute_dtype = torch.float16
      

  model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
  tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
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



  args = TrainingArguments(
      evaluation_strategy="steps",
      per_device_train_batch_size=7,
      gradient_accumulation_steps=4,
      gradient_checkpointing=True,
      learning_rate=1e-4,
      fp16 = not torch.cuda.is_bf16_supported(),
      bf16 = torch.cuda.is_bf16_supported(),
      max_steps=-1,
      num_train_epochs=3,
      save_strategy="epoch",
      logging_steps=10,
      output_dir=NEW_MODEL_NAME,
      optim="paged_adamw_32bit",
      lr_scheduler_type="linear"
  )
      

  trainer = SFTTrainer(
      model=model,
      args=args,
      train_dataset=dataset,
      dataset_text_field="text",
      max_seq_length=128,
      formatting_func=formatting_prompts_func
  )
      
  trainer.train()


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

def inference():
  set_seed(2024)

  prompt = "Africa is an emerging economy because"

  model_checkpoint = "microsoft/Phi-3-mini-4k-instruct"

  tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained(model_checkpoint,
                                              trust_remote_code=True,
                                              torch_dtype="auto",
                                              device_map="cuda")

  inputs = tokenizer(prompt,
                    return_tensors="pt").to("cuda")
  outputs = model.generate(**inputs,
                          do_sample=True, max_new_tokens=120)

  response = tokenizer.decode(outputs[0], skip_special_tokens=True)
  return response

print(inference())

"""
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
"""