# Import lainnya...
from flask import Blueprint, request, jsonify
from fastembed import TextEmbedding
from utils.authentication import authenticate
from utils.utils import calculate_token_count
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the path to the models directory
MODEL_PATH = os.getenv("MODEL_PATH", "./models")

# Initialize the embedding model with a custom path
embedding_model = TextEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    cache_dir=MODEL_PATH
)

# Create blueprint
embeddings_bp = Blueprint("embeddings", __name__)

# Load environment variables
RUNPOD_URL = os.getenv("RUNPOD_URL", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENABLE = os.getenv("RUNPOD_ENABLE", "false").lower() == "true"

@embeddings_bp.route("/v1/embeddings", methods=["POST"])
@authenticate
def embed():
    """
    Generate embeddings for the input text. Includes token calculation for input text.
    """
    data = request.get_json()
    if not data or "input" not in data:
        return jsonify({"error": "Missing 'input' field"}), 400

    # Validate input (must be string or list of strings)
    input_text = data.get("input", "")
    if not isinstance(input_text, (str, list)):
        return jsonify({"error": "Input text must be a string or list of strings"}), 400

    # Ensure input_text is always a list
    texts = input_text if isinstance(input_text, list) else [input_text]

    # Handle multiple texts
    if len(texts) > 1:
        if not RUNPOD_ENABLE:
            return jsonify({"error": "RunPod is disabled and cannot process multiple texts"}), 400

        if not RUNPOD_URL or not RUNPOD_API_KEY:
            return jsonify({"error": "RunPod URL or API key is not configured"}), 500

        try:
            # Forward the request to the external embedding service
            headers = {
                "Authorization": f"Bearer {RUNPOD_API_KEY}",
                "Content-Type": "application/json",
            }
            response = requests.post(RUNPOD_URL, json={"input": texts}, headers=headers, timeout=10)

            # Return the external service's response directly
            return jsonify(response.json()), response.status_code
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Failed to forward request: {str(e)}"}), 500

    # For a single string (or list with one item), generate embedding locally
    try:
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
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
