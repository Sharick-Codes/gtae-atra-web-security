# ============================================================
# GTAE-ATRA Cloud Security Engine Dockerfile
# Compatible with Hugging Face Spaces, Render, Railway, AWS
# ============================================================

FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    MONITOR_HOST=0.0.0.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first (lightweight, fast build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user && \
    mkdir -p /app/logs /app/data/models && \
    chown -R user:user /app

# Copy application files
COPY src/ /app/src/
COPY data/models/ /app/data/models/
COPY logs/ /app/logs/

# Switch to non-root user
USER user

# Expose server port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Start the GTAE-ATRA Web Security Server
CMD ["python", "src/web_monitor_server.py"]
