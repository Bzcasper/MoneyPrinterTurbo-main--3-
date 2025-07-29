# Use an official Python runtime as a parent image
FROM python:3.11-slim-bullseye

# Set the working directory in the container
WORKDIR /MoneyPrinterTurbo

# Set environment variables
ENV PYTHONPATH="/MoneyPrinterTurbo"
ENV PYTHONUNBUFFERED=1

# Install system dependencies and FFmpeg in a single step
RUN apt-get update && apt-get install -y \
    git \
    imagemagick \
    curl \
    gnupg \
    build-essential \
    wget \
    tar \
    xz-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

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
COPY app/requirements.txt ./requirements.txt

# Update pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy startup scripts from the correct location in the build context
COPY app/start_webui.sh /MoneyPrinterTurbo/start_webui.sh
COPY app/start_mcp_server.sh /MoneyPrinterTurbo/start_mcp_server.sh
COPY app/start_api.sh /MoneyPrinterTurbo/start_api.sh
RUN chmod +x /MoneyPrinterTurbo/*.sh

# Change ownership to appuser
RUN chown -R appuser:appuser /MoneyPrinterTurbo

# Switch to non-root user
USER appuser

# Expose ports for webui and api services
EXPOSE 8501 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command (can be overridden by docker-compose.yml)
CMD ["streamlit", "run", "./webui/Main.py", "--browser.serverAddress=0.0.0.0", "--server.enableCORS=True", "--browser.gatherUsageStats=False"]