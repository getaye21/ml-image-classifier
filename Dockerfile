FROM python:3.9-slim

# Install system dependencies - FIXED PACKAGES
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create upload directory
RUN mkdir -p /tmp/uploads

# Hugging Face uses port 7860
ENV PORT=7860

# Run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app
