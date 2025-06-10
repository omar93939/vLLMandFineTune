from unsloth import FastLanguageModel, PatchDPOTrainer
from os import environ

API_KEY = environ['PORNMIXER_HUGGINGFACE_APIKEY']

max_seq_length = 128000
dtype = None
load_in_4bit = True
model_name = "PornMixer/SFTModel"

print("model: " + model_name)

PatchDPOTrainer()

model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = model_name,
  max_seq_length = max_seq_length,
  dtype = dtype,
  load_in_4bit = load_in_4bit,
  token = API_KEY
)

from datasets import load_dataset
train = load_dataset("PornMixer/DPO_Train", split="train", token = API_KEY)
validate = load_dataset("PornMixer/DPO_Eval", split="train", token = API_KEY)

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

from trl import DPOTrainer
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported

callbacks = [
  EarlyStoppingCallback(early_stopping_patience=3)
]

trainer = DPOTrainer(
  model = model,
  tokenizer = tokenizer,
  dataset_text_field = "Creator",
  max_seq_length = max_seq_length,
  dataset_num_proc = 2,
  args = TrainingArguments(
    per_device_train_batch_size = 8,

    warmup_ratio = 0.1,
    num_train_epochs = 5,

    learning_rate = 6e-5,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 5,
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
  beta = 0.1,
  train_dataset = train,
  eval_dataset = validate,
  callbacks = callbacks
)

trainer_stats = trainer.train()

model.push_to_hub_merged("PornMixer/DPOLoRA", tokenizer, save_method = "lora", token = API_KEY)
model.push_to_hub_merged("PornMixer/DPOModel", tokenizer, save_method = "merged_16bit", token = API_KEY)
