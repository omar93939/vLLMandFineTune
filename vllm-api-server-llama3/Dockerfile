FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies, update package lists, install additional packages, clean up
RUN pip install --no-cache-dir vllm

# Expose port 8000 (http)
EXPOSE 8000

# Command to run the VLLM API server
ENTRYPOINT ["vllm", "serve"]
# Overwriteable default behavior
CMD ["PornMixer/Model", "--max_model_len", "128000", "--download-dir", "/.cache/huggingface/"]
