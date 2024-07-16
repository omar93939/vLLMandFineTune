1. 
Select GPU (7B Qwen2 needs at least `x` GB VRAM)

2. 
RunPod Pytorch 2.1 Template
  Edit Template
  Expose HTTP Port 8000

3. 
Edit Pod
  Expose HTTP Port 8000 (If it's not already by default)

2. 
SSH into the Pod (PuTTY > Runpod):
  Connect > TCP Port Mappings
  Copy `Public IP` and paste it in "Host Name (or IP address)"
  Copy `External` and paste it in "Port"
  Open
  Connect Once (or Accept, up to you :shrug:)
  Root

3. 
  pip install vllm

3.5. 
  IF long context size model (i.e. Qwen2 7b & 131072)
  

4. 
  python -m vllm.entrypoints.openai.api_server --model `model_name` *if lora* --enable-lora --lora-modules `name`=`path` `name`=`path`

  (example LoRA: 
    git clone https://huggingface.co/PornMixer/dolphin-2.9.2-qwen2-7b-LoRA
    python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-0.5B-Instruct --enable-lora --lora-modules lora=dolphin-2.9.2-qwen2-7b-LoRA
  )

5. 
Runpod Pod > Connection Options > Right click `Connect to HTTP Service` > `Copy Link Address`

1) 
  Prompt Template:
  {
    Header-Type: application/json,
    Body:
      {
        "model": `model_name`,
        "prompt": `Prompt`,
        "max_tokens": `max_tokens`,
        "temperature": `temperature (randomness)`,
        "stop": "User:"
      }
  }

  Example Prompt:
  {
    Header-Type: application/json,
    Body:
      {
        "model": "Qwen/Qwen2-7B-Instruct-GPTQ-Int4",
        "prompt": "User: Hey there! What's up?Creator: ",
        "max_tokens": 1000,
        "temperature": 0.1,
        "stop": "User:"
      }
  }
