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
# Konfigurasi Environment (tetap mempertahankan konteks aslimu)
# ======================================================================================
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# Hindari item kosong dan spasi berlebih
AVAILABLE_MODELS = [m.strip() for m in os.getenv("AVAILABLE_MODELS", "").split(",") if m.strip()]
MODEL_PATH = os.getenv("MODEL_PATH", "./models")
MAX_CACHED_MODELS = int(os.getenv("MAX_CACHED_MODELS", 1))  # Batas jumlah model di cache
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 600))            # Default 10 menit

# Forwarding - MODAL (PRIORITAS)
MODAL_URL = os.getenv("MODAL_URL", "")
MODAL_API_KEY = os.getenv("MODAL_API_KEY", "")
MODAL_ENABLE = os.getenv("MODAL_ENABLE", "false").strip().lower() in ("1","true","yes","on")

# Forwarding - RunPod (FALLBACK)
RUNPOD_URL = os.getenv("RUNPOD_URL", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENABLE = os.getenv("RUNPOD_ENABLE", "false").strip().lower() in ("1","true","yes","on")

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
        return

    try:
        from fastembed.common.model_description import PoolingType, ModelSource
    except Exception as e:
        logging.warning(
            "fastembed.common.model_description tidak tersedia (%s). "
            "Lewati registrasi custom e5.", e
        )
        if target_model in AVAILABLE_MODELS:
            AVAILABLE_MODELS = [m for m in AVAILABLE_MODELS if m != target_model]
            logging.warning(
                "Menghapus '%s' dari AVAILABLE_MODELS karena registrasi tidak tersedia.",
                target_model
            )
        return

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
# Loader model (TETAP konteks awal)
# ======================================================================================
def get_or_load_model(model_name):
    """
    Retrieve or load an embedding model. Validate against allowed models.
    """
    global LOADED_MODELS

    if model_name not in AVAILABLE_MODELS:
        logging.error(f"Requested model '{model_name}' is not in allowed models: {AVAILABLE_MODELS}")
        raise ValueError(f"Model '{model_name}' is not available. Allowed models: {AVAILABLE_MODELS}")

    if model_name in LOADED_MODELS:
        logging.info(f"Using cached model: {model_name}")
        return LOADED_MODELS[model_name]

    try:
        logging.info(f"Loading new model: {model_name}")
        model = TextEmbedding(model_name=model_name, cache_dir=MODEL_PATH)

        LOADED_MODELS[model_name] = model

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
# Helpers: normalisasi respons OpenAI + forwarders Modal/RunPod
# ======================================================================================
def _make_openai_embedding_response(model_name, embeddings_list, token_counts):
    """
    Bentuk respons OpenAI-compatible.
    embeddings_list: List[List[float]]
    token_counts:    List[int]
    """
    data_items = []
    for i, vec in enumerate(embeddings_list):
        # vec kemungkinan numpy array → pastikan list
        vlist = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        data_items.append({
            "object": "embedding",
            "index": i,
            "embedding": vlist
        })
    resp = {
        "object": "list",
        "model": model_name,
        "data": data_items,
        "usage": {
            "token_counts": token_counts,             # tambahan custom (per input)
            "prompt_tokens": int(sum(token_counts)),
            "total_tokens": int(sum(token_counts)),
        },
    }
    return resp

def _enrich_external_openai_response(resp_json, token_counts):
    """
    External (Modal/RunPod) biasanya sudah kirim OpenAI-format.
    Kita tambahkan 'usage.token_counts' (dan isi usage jika kosong).
    """
    if not isinstance(resp_json, dict):
        return resp_json

    usage = resp_json.get("usage") or {}
    usage.setdefault("prompt_tokens", int(sum(token_counts)))
    usage.setdefault("total_tokens", int(sum(token_counts)))
    usage["token_counts"] = token_counts
    resp_json["usage"] = usage
    return resp_json

def _forward_to_modal(texts, model_name):
    if not (MODAL_ENABLE and MODAL_URL and MODAL_API_KEY):
        return None, "Modal disabled or not configured"

    payload = {
        "input": {
            "openai_route": "/v1/embeddings",
            "openai_input": {
                "input": texts,
                "model": model_name
            }
        }
    }
    headers = {
        "Authorization": f"Bearer {MODAL_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        logging.info("Forwarding request to Modal.")
        r = requests.post(MODAL_URL, json=payload, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        if "output" in j and isinstance(j["output"], list) and j["output"]:
            return j["output"][0], None
        # Jika provider sudah langsung return OpenAI-style tanpa 'output'
        if "data" in j:
            return j, None
        return None, "Invalid Modal response structure"
    except Exception as e:
        return None, f"Modal forward error: {e}"

def _forward_to_runpod(texts, model_name):
    if not (RUNPOD_ENABLE and RUNPOD_URL and RUNPOD_API_KEY):
        return None, "RunPod disabled or not configured"

    payload = {
        "input": {
            "openai_route": "/v1/embeddings",
            "openai_input": {
                "input": texts,
                "model": model_name
            }
        }
    }
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        logging.info("Forwarding request to RunPod.")
        r = requests.post(RUNPOD_URL, json=payload, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        if "output" in j and isinstance(j["output"], list) and j["output"]:
            return j["output"][0], None
        if "data" in j:
            return j, None
        return None, "Invalid RunPod response structure"
    except Exception as e:
        return None, f"RunPod forward error: {e}"

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

        # Normalize texts
        texts = input_text if isinstance(input_text, list) else [input_text]

        # Hitung token_counts di awal supaya bisa disematkan terlepas dari jalur proses
        # (pakai model "gpt-4" sebagai estimator seperti sebelumnya)
        token_counts = [calculate_token_count(t, model="gpt-4") for t in texts]

        # Jika melebihi batas local → PRIORITAS Modal, fallback RunPod
        if len(texts) > MAX_TEXTS_FOR_LOCAL_PROCESSING:
            # 1) Modal
            modal_resp, modal_err = _forward_to_modal(texts, model_name)
            if modal_resp is not None:
                enriched = _enrich_external_openai_response(modal_resp, token_counts)
                return jsonify(enriched), 200
            logging.error(modal_err or "Modal forwarding failed")

            # 2) RunPod
            runpod_resp, runpod_err = _forward_to_runpod(texts, model_name)
            if runpod_resp is not None:
                enriched = _enrich_external_openai_response(runpod_resp, token_counts)
                return jsonify(enriched), 200
            logging.error(runpod_err or "RunPod forwarding failed")

            # 3) Keduanya gagal
            return jsonify({"error": "All external forwarders failed", "modal_error": modal_err, "runpod_error": runpod_err}), 502

        # Jalur local (<= batas)
        logging.info(f"Generating embeddings locally using model: {model_name}")
        embeddings = list(model.embed(texts))  # generator → list

        # Respons OpenAI-compatible + token_counts
        response = _make_openai_embedding_response(model_name, embeddings, token_counts)
        logging.info("Embeddings generated successfully (local).")
        return jsonify(response), 200

    except Exception as e:
        logging.critical(f"Unexpected error in embedding endpoint: {str(e)}")
        return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500
