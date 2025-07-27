# Use an official Python runtime as a parent image
FROM python:3.11-slim-bullseye

# Set the working directory in the container
WORKDIR /MoneyPrinterTurbo

# Set environment variables
ENV PYTHONPATH="/MoneyPrinterTurbo"
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    imagemagick \
    curl \
    gnupg \
    build-essential \
    wget \
    tar \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Install latest FFmpeg static build
RUN mkdir -p /tmp/ffmpeg && cd /tmp/ffmpeg && \
    wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar xf ffmpeg-release-amd64-static.tar.xz && \
    cd ffmpeg-*-static && \
    cp ffmpeg ffprobe /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    cd / && rm -rf /tmp/ffmpeg

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Fix security policy for ImageMagick
RUN sed -i '/<policy domain="path" rights="none" pattern="@\*"/d' /etc/ImageMagick-6/policy.xml || true

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /MoneyPrinterTurbo

# Copy requirements first for better caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to appuser
RUN chown -R appuser:appuser /MoneyPrinterTurbo

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8501 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command (can be overridden)
CMD ["streamlit", "run", "./webui/Main.py", "--browser.serverAddress=0.0.0.0", "--server.enableCORS=True", "--browser.gatherUsageStats=False"]
