from flask import Blueprint, request, jsonify
from fastembed import TextEmbedding
from utils.authentication import authenticate
from utils.utils import calculate_token_count
import os
import logging
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Konfigurasi Environment
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
AVAILABLE_MODELS = os.getenv("AVAILABLE_MODELS", "").split(",")
MODEL_PATH = os.getenv("MODEL_PATH", "./models")
MAX_CACHED_MODELS = int(os.getenv("MAX_CACHED_MODELS", 2))
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 600))
RUNPOD_URL = os.getenv("RUNPOD_URL", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENABLE = os.getenv("RUNPOD_ENABLE", "false").lower() == "true"
MAX_TEXTS_FOR_LOCAL_PROCESSING = int(os.getenv("MAX_TEXTS_FOR_LOCAL_PROCESSING", 1))

# Blueprint Flask
embeddings_bp = Blueprint("embeddings", __name__)

@embeddings_bp.route("/v1/embeddings", methods=["POST"])
@authenticate
def embed():
    """
    Generate embeddings for the input text. Supports single or batch input.
    """
    try:
        # Parse JSON input
        data = request.get_json()
        if not data or "input" not in data:
            logging.warning("Missing 'input' field in request body.")
            return jsonify({"error": "Missing 'input' field"}), 400

        input_text = data.get("input", "")
        if not isinstance(input_text, (str, list)):
            logging.warning("Invalid input type. Input must be a string or list of strings.")
            return jsonify({"error": "Input text must be a string or list of strings"}), 400

        # Ambil model dari request atau gunakan default
        model_name = data.get("model", DEFAULT_MODEL)
        if model_name not in AVAILABLE_MODELS:
            logging.error(f"Requested model '{model_name}' is not in allowed models: {AVAILABLE_MODELS}")
            return jsonify({"error": f"Model '{model_name}' is not available. Allowed models: {AVAILABLE_MODELS}"}), 400

        # Handle single or batch text input
        texts = input_text if isinstance(input_text, list) else [input_text]

        if len(texts) > MAX_TEXTS_FOR_LOCAL_PROCESSING:
            if not RUNPOD_ENABLE:
                logging.error("RunPod is disabled and cannot process multiple texts.")
                return jsonify({"error": "RunPod is disabled and cannot process multiple texts"}), 400

            if not RUNPOD_URL or not RUNPOD_API_KEY:
                logging.error("RunPod URL or API key is not configured.")
                return jsonify({"error": "RunPod URL or API key is not configured"}), 500

            try:
                # Build the payload for RunPod
                payload = {
                    "input": {
                        "openai_route": "/v1/embeddings",
                        "openai_input": {
                            "input": texts,
                            "model": model_name
                        }
                    }
                }

                # Forward the request to the external embedding service
                headers = {
                    "Authorization": f"Bearer {RUNPOD_API_KEY}",
                    "Content-Type": "application/json",
                }
                logging.info("Forwarding request to RunPod.")
                response = requests.post(RUNPOD_URL, json=payload, headers=headers, timeout=TIMEOUT)

                logging.info("Response received from RunPod.")
                runpod_response = response.json()  # Parse JSON response

                # Periksa apakah 'output' ada dalam respons
                if "output" not in runpod_response or not isinstance(runpod_response["output"], list):
                    logging.error("Invalid RunPod response structure.")
                    return jsonify({"error": "Invalid RunPod response structure"}), 500

                return jsonify(runpod_response["output"][0]), response.status_code
            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to forward request to RunPod: {str(e)}")
                return jsonify({"error": f"Failed to forward request: {str(e)}"}), 500

        # Generate embeddings locally using a context manager
        logging.info(f"Generating embeddings using model: {model_name}")
        with TextEmbedding(model_name=model_name, cache_dir=MODEL_PATH) as model:
            embeddings = list(model.embed(texts))  # Convert generator to list

        # Calculate token counts for each text
        token_counts = [calculate_token_count(text, model="gpt-4") for text in texts]

        # Format response
        response = {
            "data": [
                {
                    "object": "embedding",
                    "embedding": embeddings[i].tolist(),
                    "index": i,
                }
                for i in range(len(embeddings))
            ],
            "model": model_name,
            "usage": {
                "input_text_count": len(texts),
                "prompt_tokens": sum(token_counts),
                "total_tokens": sum(token_counts),
            },
        }
        logging.info("Embeddings generated successfully.")
        return jsonify(response)

    except Exception as e:
        logging.critical(f"Unexpected error in embedding endpoint: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500