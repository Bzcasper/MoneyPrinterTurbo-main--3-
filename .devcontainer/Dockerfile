# Use Ubuntu 22.04 as base for stability and compatibility
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    python3 \
    python3-pip \
    curl \
    build-essential \
    ffmpeg \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry (if using) or other tools
RUN pip3 install --no-cache poetry

# Set working directory
WORKDIR /workspace

# Ensure correct permissions
RUN chown -R root:root /workspace

# Default user
USER root
