# Import lainnya...
from flask import Blueprint, request, jsonify
from fastembed import TextEmbedding
from utils.authentication import authenticate
from utils.utils import calculate_token_count
import os
import requests
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file named app.log
        logging.StreamHandler()         # Also log to the console
    ]
)

# Define the path to the models directory
MODEL_PATH = os.getenv("MODEL_PATH", "./models")

# Load environment variables
RUNPOD_URL = os.getenv("RUNPOD_URL", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENABLE = os.getenv("RUNPOD_ENABLE", "false").lower() == "true"

# Ambil timeout dari environment variable, default ke 600 detik (10 menit)
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 600))

# Initialize the embedding model with a custom path
try:
    embedding_model = TextEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir=MODEL_PATH
    )
    logging.info(f"Embedding model initialized successfully. - {str(RUNPOD_ENABLE)}")
except Exception as e:
    logging.error(f"Failed to initialize the embedding model: {str(e)}")
    raise e

# Create blueprint
embeddings_bp = Blueprint("embeddings", __name__)

@embeddings_bp.route("/v1/embeddings", methods=["POST"])
@authenticate
def embed():
    """
    Generate embeddings for the input text. Includes token calculation for input text.
    """
    try:
        # Parse JSON input
        data = request.get_json()
        if not data or "input" not in data:
            logging.warning("Missing 'input' field in request body.")
            return jsonify({"error": "Missing 'input' field"}), 400

        # Validate input (must be string or list of strings)
        input_text = data.get("input", "")
        if not isinstance(input_text, (str, list)):
            logging.warning("Invalid input type. Input must be a string or list of strings.")
            return jsonify({"error": "Input text must be a string or list of strings"}), 400

        # Ensure input_text is always a list
        texts = input_text if isinstance(input_text, list) else [input_text]

        # Handle multiple texts
        if len(texts) > 1:
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
                            "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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
                # logging.debug(f"RunPod response: {runpod_response}")

                # Periksa apakah 'output' ada dalam respons
                if "output" not in runpod_response or not isinstance(runpod_response["output"], list):
                    logging.error("Invalid RunPod response structure.")
                    return jsonify({"error": "Invalid RunPod response structure"}), 500

                # Ambil elemen pertama dari 'output' dan kembalikan sebagai respons
                return jsonify(runpod_response["output"][0]), response.status_code
                # return jsonify(response.output[0].json()), response.status_code


            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to forward request to RunPod: {str(e)}")
                return jsonify({"error": f"Failed to forward request: {str(e)}"}), 500

        # For a single string (or list with one item), generate embedding locally
        try:
            logging.info("Generating embedding locally.")
            # Calculate token count
            token_count = calculate_token_count(texts[0], model="gpt-4")

            # Generate embeddings (convert generator to list)
            embeddings = list(embedding_model.embed(texts))

            # Format the response
            response = {
                "data": [
                    {
                        "object": "embedding",
                        "embedding": embeddings[0].tolist(),  # Access first embedding
                        "index": 0,
                        "tokens_used": token_count,
                    }
                ],
                "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "usage": {
                    "input_tokens": token_count,
                    "total_tokens": token_count,
                },
            }
            logging.info("Embedding generated successfully.")
            return jsonify(response)
        except Exception as e:
            logging.error(f"Error generating embedding locally: {str(e)}")
            return jsonify({"error": str(e)}), 500

    except Exception as e:
        logging.critical(f"Unexpected error in embedding endpoint: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500
