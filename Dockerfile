FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better Docker layer caching)
COPY requirements_runpod.txt /requirements.txt

# Configure pip to be more tolerant of large downloads
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=5

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy handler
COPY handler_runpod.py /handler.py

# Set working directory
WORKDIR /

# Expose port
EXPOSE 8000

# Start the RunPod handler (which launches vLLM as needed)
CMD ["python", "-u", "/handler.py"]
