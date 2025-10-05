# routes/embeddings.py
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

# ======================================================================================
# Konfigurasi Environment (TETAP: mempertahankan konteks aslimu)
# ======================================================================================
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# Hindari item kosong dan spasi berlebih
AVAILABLE_MODELS = [m.strip() for m in os.getenv("AVAILABLE_MODELS", "").split(",") if m.strip()]
MODEL_PATH = os.getenv("MODEL_PATH", "./models")
MAX_CACHED_MODELS = int(os.getenv("MAX_CACHED_MODELS", 1))  # Batas jumlah model di cache
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 600))            # Default 10 menit
RUNPOD_URL = os.getenv("RUNPOD_URL", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENABLE = os.getenv("RUNPOD_ENABLE", "false").lower() == "true"
MAX_TEXTS_FOR_LOCAL_PROCESSING = int(os.getenv("MAX_TEXTS_FOR_LOCAL_PROCESSING", 1))

# Cache model yang dimuat
LOADED_MODELS = {}

# ======================================================================================
# Tambahan: Registrasi Custom Model untuk FastEmbed (menjaga konteks, hanya menambah blok)
# ======================================================================================
def _register_custom_e5_and_sanitize_available_models():
    """
    - Coba daftarkan 'intfloat/multilingual-e5-base' sebagai custom model FastEmbed.
    - Jika registrasi tidak didukung (versi fastembed lama), dan model itu tercantum
      di AVAILABLE_MODELS, buang dari AVAILABLE_MODELS agar validasi startup tidak gagal.
    """
    global AVAILABLE_MODELS

    target_model = "intfloat/multilingual-e5-base"
    needs_e5 = any(m == target_model for m in AVAILABLE_MODELS) or (DEFAULT_MODEL == target_model)

    if not needs_e5:
        # Tidak perlu registrasi jika tidak dipakai.
        return

    try:
        # Kelas-kelas ini tersedia di fastembed versi tertentu.
        from fastembed.common.model_description import PoolingType, ModelSource
    except Exception as e:
        logging.warning(
            "fastembed.common.model_description tidak tersedia (%s). "
            "Lewati registrasi custom e5.", e
        )
        # Agar validate_models tidak gagal, hapus dari daftar jika ada
        if target_model in AVAILABLE_MODELS:
            AVAILABLE_MODELS = [m for m in AVAILABLE_MODELS if m != target_model]
            logging.warning(
                "Menghapus '%s' dari AVAILABLE_MODELS karena registrasi tidak tersedia.",
                target_model
            )
        # Jika DEFAULT_MODEL adalah e5, biarkan validation nanti yang mengangkat error jelas
        return

    # Cek apakah add_custom_model ada
    if not hasattr(TextEmbedding, "add_custom_model"):
        logging.warning(
            "FastEmbed tidak mendukung add_custom_model() pada versi saat ini. "
            "Lewati registrasi custom e5."
        )
        if target_model in AVAILABLE_MODELS:
            AVAILABLE_MODELS = [m for m in AVAILABLE_MODELS if m != target_model]
            logging.warning(
                "Menghapus '%s' dari AVAILABLE_MODELS karena add_custom_model() tidak ada.",
                target_model
            )
        return

    try:
        # Registrasi model custom e5
        TextEmbedding.add_custom_model(
            model=target_model,
            pooling=PoolingType.MEAN,                # e5 pakai mean pooling
            normalization=True,                      # cosine-ready
            sources=ModelSource(hf=target_model),
            model_file="onnx/model.onnx",            # path ONNX pada repo HF
            dim=768,
        )
        logging.info("Custom model '%s' terdaftar di FastEmbed.", target_model)
    except Exception as e:
        logging.warning("Gagal register custom model e5: %s", e)
        # Untuk mencegah crash saat validasi, buang dari AVAILABLE_MODELS jika ada
        if target_model in AVAILABLE_MODELS:
            AVAILABLE_MODELS = [m for m in AVAILABLE_MODELS if m != target_model]
            logging.warning(
                "Menghapus '%s' dari AVAILABLE_MODELS karena registrasi gagal.",
                target_model
            )

# PANGGIL registrasi custom SEBELUM validasi model/instansiasi TextEmbedding
_register_custom_e5_and_sanitize_available_models()

# ======================================================================================
# Fungsi validasi model pada startup (TETAP sesuai konteks awal)
# ======================================================================================
def validate_models(available_models, model_path):
    """
    Validates all available models at application startup.
    """
    for model_name in available_models:
        try:
            logging.info(f"Validating model '{model_name}'...")
            _ = TextEmbedding(model_name=model_name, cache_dir=model_path)
            logging.info(f"Model '{model_name}' is valid and ready.")
        except Exception as e:
            logging.error(f"Model '{model_name}' cannot be loaded: {str(e)}")
            raise ValueError(f"Invalid model '{model_name}' in AVAILABLE_MODELS: {str(e)}")

# Validasi model pada startup (TETAP: akan raise jika ada yang invalid)
try:
    validate_models(AVAILABLE_MODELS, MODEL_PATH)
except ValueError as e:
    logging.critical(f"Model validation failed: {str(e)}")
    raise e

# ======================================================================================
# Loader model (TETAP konteks awal, hanya variabel globalnya yang sama)
# ======================================================================================
def get_or_load_model(model_name):
    """
    Retrieve or load an embedding model. Validate against allowed models.
    """
    global LOADED_MODELS

    # Validasi apakah model termasuk dalam daftar model yang diizinkan
    if model_name not in AVAILABLE_MODELS:
        logging.error(f"Requested model '{model_name}' is not in allowed models: {AVAILABLE_MODELS}")
        raise ValueError(f"Model '{model_name}' is not available. Allowed models: {AVAILABLE_MODELS}")

    # Jika model ada di cache, gunakan model tersebut
    if model_name in LOADED_MODELS:
        logging.info(f"Using cached model: {model_name}")
        return LOADED_MODELS[model_name]

    # Jika tidak ada, muat model baru
    try:
        logging.info(f"Loading new model: {model_name}")
        model = TextEmbedding(model_name=model_name, cache_dir=MODEL_PATH)

        # Tambahkan ke cache
        LOADED_MODELS[model_name] = model

        # Hapus model lama jika cache penuh
        if len(LOADED_MODELS) > MAX_CACHED_MODELS:
            oldest_model = next(iter(LOADED_MODELS))
            if oldest_model != model_name:
                del LOADED_MODELS[oldest_model]
                logging.warning(f"Removed oldest model from cache: {oldest_model}")

        return model
    except Exception as e:
        logging.error(f"Failed to load model '{model_name}': {str(e)}")
        raise Exception(f"Failed to load model '{model_name}': {str(e)}")

# ======================================================================================
# Blueprint Flask (TETAP konteks awal)
# ======================================================================================
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
        try:
            model = get_or_load_model(model_name)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        # Handle single atau batch
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

                # Forward request ke RunPod
                headers = {
                    "Authorization": f"Bearer {RUNPOD_API_KEY}",
                    "Content-Type": "application/json",
                }
                logging.info("Forwarding request to RunPod.")
                response = requests.post(RUNPOD_URL, json=payload, headers=headers, timeout=TIMEOUT)

                logging.info("Response received from RunPod.")
                runpod_response = response.json()  # Parse JSON response

                # Validasi struktur response
                if "output" not in runpod_response or not isinstance(runpod_response["output"], list):
                    logging.error("Invalid RunPod response structure.")
                    return jsonify({"error": "Invalid RunPod response structure"}), 500

                # Ambil elemen pertama
                return jsonify(runpod_response["output"][0]), response.status_code
            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to forward request to RunPod: {str(e)}")
                return jsonify({"error": f"Failed to forward request: {str(e)}"}), 500

        # Generate embeddings lokal
        logging.info(f"Generating embeddings using model: {model_name}")
        embeddings = list(model.embed(texts))  # Convert generator to list

        # Hitung token count per text
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
