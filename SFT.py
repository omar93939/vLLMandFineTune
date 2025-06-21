from unsloth import FastLanguageModel
from os import environ
from torch import bfloat16

API_KEY = environ['HF_TOKEN']

# Can increase this with more VRAM (https://github.com/unslothai/unsloth?tab=readme-ov-file#llama-31-8b-max-context-length)
max_seq_length = 128000
dtype = bfloat16
load_in_4bit = True
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

print("model: " + model_name)

model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = model_name,
  max_seq_length = max_seq_length,
  dtype = dtype,
  load_in_4bit = load_in_4bit
)

from datasets import load_dataset
data_files = {
  "train": "SFT_train.jsonl",
  "validation": "SFT_validation.jsonl"
}

train = load_dataset("PornMixer/Dataset", data_files=data_files, split="train", token = API_KEY)
validate = load_dataset("PornMixer/Dataset", data_files=data_files, split="validation", token = API_KEY)

train, validate = train.rename_column("Creator", "text"), validate.rename_column("Creator", "text")

print(train)
print(validate)

model = FastLanguageModel.get_peft_model(
  model,
  r = 16,
  target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
  lora_alpha = 16,
  lora_dropout = 0,
  bias = "none",
  use_gradient_checkpointing = "unsloth",
  random_state = 3407,
  use_rslora = False,
  loftq_config = None,
  max_seq_length = max_seq_length
)

from trl import SFTTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported

callbacks = [
  EarlyStoppingCallback(early_stopping_patience=3)
]

trainer = SFTTrainer(
  model = model,
  tokenizer = tokenizer,
  max_seq_length = max_seq_length,
  dataset_num_proc = 2,
  args = TrainingArguments(
    # Test this below
    per_device_train_batch_size = 16,

    warmup_ratio = 0.1,
    num_train_epochs = 3,

    learning_rate = 3e-5,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 10,
    optim = "adamw_8bit",
    weight_decay = 0.0001,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = "outputs",
    eval_strategy = "steps",
    eval_steps = 10,
    save_steps = 10,
    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss"
  ),
  train_dataset = train,
  eval_dataset = validate,
  callbacks = callbacks
)

trainer_stats = trainer.train()
if trainer_stats is not None:
  print("Training stats:", trainer_stats)

model.push_to_hub_merged("PornMixer/SFTLoRA", tokenizer, save_method = "lora", token = API_KEY)
model.push_to_hub_merged("PornMixer/SFTModel", tokenizer, save_method = "merged_16bit", token = API_KEY)
