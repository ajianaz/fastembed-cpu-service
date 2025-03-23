# FastEmbed CPU Service

## Overview

FastEmbed CPU Service is a lightweight service designed for generating text embeddings efficiently on CPU environments. The service is built using Flask, integrates FastEmbed for embedding generation, and supports token calculations using OpenAI's `tiktoken` library.

## Features

- Generate embeddings for text inputs using the `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model.
- Forward requests to external services (e.g., RunPod) for batch processing when enabled.
- Token calculation endpoint leveraging OpenAI's `tiktoken` library with support for `gpt-4` models.
- Configurable via environment variables.
- Multi-platform support with Docker (AMD64, ARMv8, etc.).

## Requirements

- Python 3.9+
- Docker (optional, for containerized deployment)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ajianaz/fastembed-cpu-service.git
cd fastembed-cpu-service
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```env
# General settings
MODEL_PATH=./models
API_KEYS=your_api_key1,your_api_key2

# RunPod settings
RUNPOD_URL=https://api.runpod.io
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENABLE=false

# Gunicorn settings
GUNICORN_WORKERS=2

# Token calculation settings
DEFAULT_MODEL=gpt-4
```

### 4. Run the Service Locally

Run the service using Gunicorn:

```bash
gunicorn -w 2 -b 0.0.0.0:5005 app:app
```

## Usage

### Endpoints

#### 1. Generate Embeddings

**Endpoint:** `/v1/embeddings`
**Method:** `POST`
**Headers:**

- `Authorization: Bearer <API_KEY>`

**Request Body:**

```json
{
  "input": "This is a test sentence."
}
```

**Response:**

```json
{
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, 0.456, ...],
      "index": 0
    }
  ],
  "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
  "usage": {
    "input_tokens": 6,
    "total_tokens": 6
  }
}
```

#### 2. Calculate Tokens

**Endpoint:** `/v1/calculate-tokens`
**Method:** `POST`
**Request Body:**

```json
{
  "text": "This is a test sentence.",
  "model": "gpt-4"
}
```

**Response:**

```json
{
  "tokens": 6
}
```

## Docker Deployment

### 1. Build and Run the Docker Image

```bash
docker build -t fastembed-cpu-service:latest .
docker run -p 5005:5005 --env-file .env fastembed-cpu-service:latest
```

### 2. Using Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  fastembed:
    build: .
    ports:
      - '5005:5005'
    env_file:
      - .env
```

Run the service:

```bash
docker-compose up --build
```

## Multi-Platform Docker Build

Use the following command to build and push multi-platform Docker images:

```bash
docker buildx build --platform linux/amd64,linux/arm64 -t <your-dockerhub-username>/fastembed-cpu-service:latest . --push
```

## GitHub Actions for CI/CD

The repository includes a GitHub Actions workflow to build and push the Docker image to GitHub Container Registry. Update the `GHCR_PAT` secret and ensure the workflow file references the correct image name (`fastembed-cpu-service`).

## Notes

- Ensure the `MODEL_PATH` directory exists and is writable for caching models.
- When using RunPod, set `RUNPOD_ENABLE` to `true` and provide valid `RUNPOD_URL` and `RUNPOD_API_KEY` values.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
