# Dockerfile
FROM python:3.11-slim

# Install system dependencies for dlib & OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy code and install dependencies
COPY . /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "image_recognition.py"]
