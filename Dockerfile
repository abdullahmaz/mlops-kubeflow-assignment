# Base Python image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if needed later for ML / KFP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code (updated in later tasks)
COPY . .

CMD ["python", "src/model_training.py"]


