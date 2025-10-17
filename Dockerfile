FROM python:3.9-slim

WORKDIR /app

# Install system dependencies, INCLUDING git and git-lfs
# This is the fix for the "git: 'lfs' is not a git command" error
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

# Install dependencies (your optimized layer caching is good)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy==1.24.3" "Cython<3.0" && \
    pip install --no-cache-dir --no-build-isolation "scikit-surprise==1.1.3" && \
    pip install --no-cache-dir -r requirements.txt

COPY . /app/

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Gunicorn with optimized settings for Railway
CMD ["gunicorn", "-w", "1", "--preload", "--timeout", "300", \
    "--worker-tmp-dir", "/dev/shm", \
    "--max-requests", "100", "--max-requests-jitter", "10", \
    "--bind", "0.0.0.0:8000", "app:app"]

