# Use a base image with Python
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies, Update package lists, install additional packages, clean up, Clone the repository and set up Git LFS
RUN apt-get update && \
    apt-get install -y \
    git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install vllm

EXPOSE 8000
EXPOSE 22

# Command to run the VLLM API server
CMD ["sh", "-c", "git clone https://huggingface.co/PornMixer/lora && python -m vllm.entrypoints.openai.api_server --max_model_len 16000 --model cognitivecomputations/dolphin-2.9.2-qwen2-7b --enable-lora --lora-modules lora=lora"]
