FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better Docker layer caching)
COPY requirements_runpod.txt /requirements.txt

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy handler
COPY handler_runpod.py /handler.py

# Set working directory
WORKDIR /

# Expose port
EXPOSE 8000

# Start vLLM server
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "datalab-to/chandra", \
     "--trust-remote-code", \
     "--port", "8000", \
     "--host", "0.0.0.0"]
