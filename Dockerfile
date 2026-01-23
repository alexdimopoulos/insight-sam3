# INSIGHT - Indoor Scene Intelligence Pipeline
# Docker container with CUDA support for SAM3-based 3D segmentation

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libopenexr-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip

# Set working directory
WORKDIR /usr/src/app

# Install Python dependencies
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create mount point directories
RUN mkdir -p /datasets \
             /usr/src/app/model_weights \
             /usr/src/app/output_results

# Define volumes for data persistence
VOLUME /datasets
VOLUME /usr/src/app/model_weights
VOLUME /usr/src/app/output_results

# Set entrypoint to run.py (arguments passed via docker run)
ENTRYPOINT ["python3.11", "run.py"]

# Default arguments (can be overridden)
CMD ["--help"]
