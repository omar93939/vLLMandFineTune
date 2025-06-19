# Stage 1: Builder
# Use a CUDA 'devel' image as the builder. This ensures that the build environment
# has the same CUDA version and OS (Ubuntu 22.04) as the final image.
# The 'devel' image includes compilers and headers (like nvcc) needed to build
# GPU-accelerated Python packages correctly.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

# Set ARG for Python version for easy configuration.
ARG PYTHON_VERSION=3.10

# Prevents apt-get from hanging on interactive prompts.
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, the virtual environment module (venv), and other build essentials.
# Running on Ubuntu 22.04, these packages are readily available.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    build-essential \
    cmake \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment to encapsulate dependencies.
RUN python${PYTHON_VERSION} -m venv /opt/venv

# Activate the virtual environment for all subsequent commands in this stage.
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install vllm.
# --no-cache-dir reduces layer size. Because this stage has CUDA, pip will
# download the correct GPU-enabled wheels for PyTorch and other dependencies.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir vllm

# ---

# Stage 2: Final Image
# Use the 'base' CUDA image for the final stage. It's much smaller than the
# 'devel' image because it doesn't include build tools.
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Copy the virtual environment from the builder stage.
# This is much faster and more reliable than reinstalling packages.
COPY --from=builder /opt/venv /opt/venv

# Set the working directory.
WORKDIR /app

# Add the virtual environment's bin directory to the PATH.
ENV PATH="/opt/venv/bin:$PATH"

# Expose the application port.
EXPOSE 8000

# Set the entrypoint to the vllm command-line tool.
# Using exec form of ENTRYPOINT is recommended.
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]

# Set the default command-line arguments for the service.
# These can be overridden when running the container.
CMD ["--model", "PornMixer/Model", "--max-model-len", "128000", "--download-dir", "/.cache/huggingface/"]
