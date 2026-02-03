# Deployment Guide
## Alzheimer's Detection Web Application

---

## Table of Contents

1. [Local Deployment](#1-local-deployment)
2. [Docker Deployment](#2-docker-deployment)
3. [Cloud Deployment](#3-cloud-deployment)
4. [Production Considerations](#4-production-considerations)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. Local Deployment

### 1.1 Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM
- 2GB free disk space

### 1.2 Step-by-Step Setup

#### Step 1: Download Project

```bash
# Clone or extract project
cd /path/to/alzheimers-detection-project
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
cd app
pip install -r requirements.txt
```

#### Step 4: Prepare Model

```bash
# Create models directory
mkdir -p models

# Copy your trained model
cp /path/to/your/alzheimer_cnn_model.h5 models/
```

#### Step 5: Run Application

```bash
# Run with model
python app.py ../models/alzheimer_cnn_model.h5

# Or run in demo mode (no model required)
python app.py
```

#### Step 6: Access Application

Open browser and navigate to: **http://localhost:5000**

---

## 2. Docker Deployment

### 2.1 Create Dockerfile

Create `Dockerfile` in project root:

```dockerfile
# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ .
COPY model/ ./model/
COPY preprocessing/ ./preprocessing/
COPY explainability/ ./explainability/
COPY utils/ ./utils/

# Create necessary directories
RUN mkdir -p static/uploads static/results

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Run application
CMD ["python", "app.py"]
```

### 2.2 Build and Run

```bash
# Build Docker image
docker build -t alzheimers-detection .

# Run container
docker run -p 5000:5000 alzheimers-detection

# Run with mounted model volume
docker run -p 5000:5000 -v /path/to/models:/app/models alzheimers-detection

# Run in detached mode
docker run -d -p 5000:5000 --name alzheimers-app alzheimers-detection
```

### 2.3 Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  alzheimers-app:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./app/static/uploads:/app/static/uploads
      - ./app/static/results:/app/static/results
    environment:
      - FLASK_ENV=production
      - MODEL_PATH=/app/models/alzheimer_cnn_model.h5
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run with Docker Compose:

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 3. Cloud Deployment

### 3.1 AWS Deployment

#### Option A: AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize EB
eb init -p python-3.9 alzheimers-detection

# Create environment
eb create alzheimers-env

# Deploy
eb deploy

# Open application
eb open
```

#### Option B: AWS EC2

1. **Launch EC2 Instance:**
   - AMI: Ubuntu 20.04 LTS
   - Instance Type: t3.medium (or larger)
   - Security Group: Allow port 5000

2. **Setup Instance:**

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Clone project
git clone https://github.com/yourusername/alzheimers-detection.git
cd alzheimers-detection

# Build and run
docker-compose up -d
```

3. **Configure Nginx (Optional):**

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3.2 Google Cloud Platform

#### App Engine Deployment

Create `app.yaml`:

```yaml
runtime: python39

instance_class: F4

automatic_scaling:
  target_cpu_utilization: 0.65
  min_instances: 1
  max_instances: 10

env_variables:
  MODEL_PATH: "gs://your-bucket/models/alzheimer_cnn_model.h5"

handlers:
- url: /static
  static_dir: app/static

- url: /.*
  script: auto
```

Deploy:

```bash
# Install Cloud SDK
gcloud components install app-engine-python

# Deploy
gcloud app deploy

# View logs
gcloud app logs tail -s default
```

### 3.3 Microsoft Azure

#### App Service Deployment

```bash
# Login to Azure
az login

# Create resource group
az group create --name alzheimers-rg --location eastus

# Create app service plan
az appservice plan create --name alzheimers-plan --resource-group alzheimers-rg --sku B1 --is-linux

# Create web app
az webapp create --resource-group alzheimers-rg --plan alzheimers-plan --name alzheimers-detection --runtime "PYTHON|3.9"

# Configure deployment
git remote add azure https://alzheimers-detection.scm.azurewebsites.net:443/alzheimers-detection.git

# Deploy
git push azure main
```

---

## 4. Production Considerations

### 4.1 Security

```python
# app.py - Security configurations
from flask import Flask
from flask_talisman import Talisman

app = Flask(__name__)

# Enable HTTPS-only
Talisman(app, force_https=True)

# Secret key
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')

# File upload security
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
```

### 4.2 Performance Optimization

```python
# Model optimization
def optimize_model(model):
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Save
    with open('model_optimized.tflite', 'wb') as f:
        f.write(tflite_model)
    
    return tflite_model
```

### 4.3 Monitoring

```python
# Add monitoring middleware
import time
import logging

@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    elapsed = time.time() - g.start_time
    app.logger.info(f'{request.method} {request.path} - {elapsed:.2f}s')
    return response
```

### 4.4 Environment Variables

Create `.env` file:

```bash
# .env
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
MODEL_PATH=/app/models/alzheimer_cnn_model.h5
MAX_CONTENT_LENGTH=16777216
UPLOAD_FOLDER=/app/static/uploads
RESULTS_FOLDER=/app/static/results
```

Load in app:

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 5. Troubleshooting

### Issue 1: Model Not Loading

**Symptom:** App runs but predictions are simulated

**Solution:**
```bash
# Check model path
ls -la models/

# Verify model format
python -c "import tensorflow as tf; print(tf.keras.models.load_model('models/model.h5'))"

# Check logs
python app.py 2>&1 | grep -i error
```

### Issue 2: Out of Memory

**Symptom:** `ResourceExhaustedError` or process killed

**Solution:**
```python
# Limit TensorFlow GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
    )
```

### Issue 3: Slow Predictions

**Symptom:** Inference takes >5 seconds

**Solutions:**
1. Use GPU if available
2. Batch predictions
3. Use TensorFlow Lite
4. Enable model caching

```python
# Cache model in memory
_model = None

def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model
```

### Issue 4: File Upload Errors

**Symptom:** "File too large" or "Invalid file type"

**Solution:**
```python
# Check file size before upload
if request.content_length > app.config['MAX_CONTENT_LENGTH']:
    return jsonify({'error': 'File too large'}), 413

# Validate extension
if not allowed_file(file.filename):
    return jsonify({'error': 'Invalid file type'}), 400
```

### Issue 5: Grad-CAM Not Working

**Symptom:** No heatmap generated

**Solution:**
```bash
# Check OpenCV installation
python -c "import cv2; print(cv2.__version__)"

# Verify model has conv layers
python -c "
import tensorflow as tf
model = tf.keras.models.load_model('models/model.h5')
for layer in model.layers:
    if 'conv' in layer.name:
        print(layer.name)
"
```

---

## Quick Reference

### Commands Summary

```bash
# Local development
python app.py

# Docker
docker build -t alzheimers-detection .
docker run -p 5000:5000 alzheimers-detection

# Docker Compose
docker-compose up -d

# AWS EB
eb deploy

# GCP
gcloud app deploy

# Azure
az webapp up --sku B1
```

### Port Configuration

| Environment | Port | URL |
|-------------|------|-----|
| Local | 5000 | http://localhost:5000 |
| Docker | 5000 | http://localhost:5000 |
| AWS | 80/443 | http://your-domain.com |
| GCP | 8080 | https://your-project.appspot.com |
| Azure | 80/443 | https://your-app.azurewebsites.net |

---

## Support

For deployment issues:
1. Check application logs
2. Verify environment variables
3. Test locally before deploying
4. Review cloud provider documentation

---

**Happy Deploying! 🚀**
