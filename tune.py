from unsloth import FastLanguageModel
import torch

max_seq_length = 131072
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = "Qwen/Qwen2-7B-Instruct-GPTQ-Int4",
  max_seq_length = max_seq_length,
  dtype = dtype,
  load_in_4bit = load_in_4bit
)

from datasets import load_dataset
dataset = load_dataset("PornMixer/ExpectedGeneration", split="train")

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
  train_dataset = dataset,
  dataset_text_field = "Creator",
  max_seq_length = max_seq_length,
  dataset_num_proc = 2,
  args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps= 4,

    warmup_steps = 5,
    max_steps = 60,

    learning_rate = 2e-4,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = "outputs",
  )
)

trainer_stats = trainer.train()

model.push_to_hub("PornMixer/dolphin-2.9.2-qwen2-7b-LoRA", token="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")
tokenizer.push_to_hub("PornMixer/dolphin-2.9.2-qwen2-7b-LoRA", token="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")
