from unsloth import FastLanguageModel
import torch
import gc

max_seq_length = 128000
dtype = None
load_in_4bit = True
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

print("model name: " + model_name)

model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = model_name,
  max_seq_length = max_seq_length,
  dtype = dtype,
  load_in_4bit = load_in_4bit
)

from datasets import load_dataset
train = load_dataset("PornMixer/ExpectedGeneration", split="train")
validate = load_dataset("PornMixer/ValidationGeneration", split="train")

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
  loftq_config = None
)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
  model = model,
  tokenizer = tokenizer,
  dataset_text_field = "Creator",
  max_seq_length = max_seq_length,
  dataset_num_proc = 2,
  args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps= 4,

    warmup_steps = 5,
    num_train_epochs = 10,

    learning_rate = 2e-4,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 10,
    optim = "adamw_8bit",
    weight_decay = 0.01,
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
)

trainer_stats = trainer.train()

model.push_to_hub("PornMixer/LoRA", token="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")
tokenizer.push_to_hub("PornMixer/LoRA", token="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")

lora_adapter_path = "lora_adapter"
model.save_pretrained(lora_adapter_path, save_adapter=True, save_config=True)

del model
gc.collect()
if torch.cuda.is_available():
  torch.cuda.empty_cache()

merged_model = FastLanguageModel.from_pretrained(
  model_name = model_name,
  max_seq_length = max_seq_length
)
merged_model = merged_model.merge_and_unload(lora_adapter_path)

merged_model.push_to_hub("PornMixer/Model", token="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")
tokenizer.push_to_hub("PornMixer/Model", token="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")
