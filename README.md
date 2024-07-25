1. 
Select GPU (7B Qwen2 needs at least `x` GB VRAM)

2. 
RunPod Pytorch 2.1 Template
  Edit Template
  Expose HTTP Port 8000

3. 
SSH into the Pod (PuTTY > Runpod):
  Connect > TCP Port Mappings
  Copy `Public IP` and paste it in "Host Name (or IP address)"
  Copy `External` and paste it in "Port"
  Open
  Connect Once (or Accept, up to you :shrug:)
  Root

4. 
  pip install vllm

5. 
  python -m vllm.entrypoints.openai.api_server --model `model_name` *if lora* --enable-lora --lora-modules `name`=`path` `name`=`path`

  (example LoRA: 
    apt update
    apt install git-lfs
    git clone https://huggingface.co/PornMixer/lora
    python -m vllm.entrypoints.openai.api_server --max_model_len 16000 --model cognitivecomputations/dolphin-2.9.2-qwen2-7b --enable-lora --lora-modules lora=lora
  )

6. 
Runpod Pod > Connection Options > Right click `Connect to HTTP Service` > `Copy Link Address`

7. 
Send prompt to `link`v1/completions



1) 
  Prompt Template:
  {
    Header-Type: application/json,
    Body:
      {
        "model": `model_name (LoRA name if applicable)`,
        "prompt": `Prompt`,
        "max_tokens": `max_tokens`,
        "temperature": `temperature (randomness)`,
        "stop": "User:"
      }
  }

  Example Prompt:
  curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "lora",
        "prompt": "USER:hey, whats your name?CREATORRESPONSE:",
        "max_tokens": 16000,
        "temperature": 0.1,
        "stop": ["SYSTEM:", "USER:", "CREATOR:", "CREATORRESPONSE:", "USERRESPONSE:"]
    }'

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
