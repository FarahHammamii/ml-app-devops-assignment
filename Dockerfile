# Use official Python runtime as base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy all application code to container
COPY . .

# Create models directory for storing trained models
RUN mkdir -p models

# Run Training
CMD ["python", "src/train.py"]
