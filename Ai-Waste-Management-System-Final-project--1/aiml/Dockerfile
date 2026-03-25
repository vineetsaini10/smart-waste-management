# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Assuming docker build context is the aiml folder
COPY requirements.txt /app/aiml/requirements.txt
RUN pip install --no-cache-dir -r /app/aiml/requirements.txt

COPY . /app/aiml

# Ensure absolute imports like 'from aiml.utils...' work
ENV PYTHONPATH="/app"

EXPOSE 8000

CMD ["uvicorn", "aiml.main:app", "--host", "0.0.0.0", "--port", "8000"]
