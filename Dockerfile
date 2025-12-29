# Use official PyTorch image with CUDA 12.8 for Blackwell (B200) support
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY model_server.py worker.py start_server.sh /app/

# Create log directory for Vast.ai PyWorker
RUN mkdir -p /var/log/model

# Make start script executable
RUN chmod +x /app/start_server.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Expose port for model server
EXPOSE 18000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:18000/health || exit 1

# Run the model server (Vast.ai will run worker.py separately via PYWORKER_REPO)
CMD ["bash", "start_server.sh"]
