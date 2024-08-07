from unsloth import FastLanguageModel

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
    gradient_accumulation_steps = 4,

    warmup_steps = 5,
    num_train_epochs = 5,

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

model.push_to_hub_merged("PornMixer/LoRA", tokenizer, save_method = "lora", token = "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")
model.push_to_hub_merged("PornMixer/Model", tokenizer, save_method = "merged_16bit", token = "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")
