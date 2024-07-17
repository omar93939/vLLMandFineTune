from unsloth import FastLanguageModel
import torch

max_seq_length = 16000
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = "cognitivecomputations/dolphin-2.9.2-qwen2-7b",
  max_seq_length = max_seq_length,
  dtype = dtype,
  load_in_4bit = load_in_4bit
)

from datasets import load_dataset
train = load_dataset("PornMixer/ExpectedGeneration", split="train")
validate = load_dataset("PornMixer/ValidationGeneration", split="train")
test = load_dataset("PornMixer/TestGeneration", split="train")

print(train)
print(validate)
print(test)

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
  train_dataset = train,
  eval_dataset = validate,
  dataset_text_field = "Creator",
  max_seq_length = max_seq_length,
  dataset_num_proc = 2,
  args = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps= 4,

    warmup_steps = 5,
    num_train_epochs = 1,

    learning_rate = 2e-4,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 10,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = "outputs",
    evaluation_strategy = "steps",
    eval_steps = 10,
    save_steps = 10,
    load_best_model_at_end = True,
    metric_for_best_model = "eval_loss"
  )
)

trainer_stats = trainer.train()

test_results = trainer.evaluate(eval_dataset = test)

model.push_to_hub("PornMixer/dolphin-2.9.2-qwen2-7b-LoRA", token="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")
tokenizer.push_to_hub("PornMixer/dolphin-2.9.2-qwen2-7b-LoRA", token="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")

print("Test results:", test_results)
