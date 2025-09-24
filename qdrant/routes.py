"""
qdrant/routes.py
Blueprint Flask utk endpoint Qdrant (CRUD koleksi & vektor).
- Tetap sederhana, sesuai konteks awal.
- Auth dibiarkan fleksibel: jika proyekmu ada decorator @authenticate, aktifkan; jika tidak, gunakan NoAuth.
"""

from flask import Blueprint, request, jsonify
from .utils import (
    create_collection, get_all_collections, get_collection_info, delete_collection,
    save_vector, save_vectors_batch, search_vector, delete_vector_by_id,
    get_vectors, set_payload, clear_payload, update_vector
)
from .config import DEFAULT_COLLECTION

# --- Auth decorator opsional ---
try:
    # Jika proyek punya auth sendiri, pakai ini:
    from utils.authentication import authenticate  # type: ignore
except Exception:
    # Fallback: no-op decorator (biar file ini independen)
    def authenticate(fn):
        return fn

qdrant_bp = Blueprint("qdrant", __name__)

# -------- Collection endpoints --------
@qdrant_bp.route("/collection/create", methods=["POST"])
@authenticate
def create_collection_route():
    try:
        data = request.json or {}
        name = data.get("collection_name")
        size = data.get("vector_size")
        distance = data.get("distance")
        vectors = data.get("vectors")  # named vectors config

        if not name:
            return jsonify({"success": False, "message": "collection_name is required"}), 400
        if not vectors and not size:
            return jsonify({"success": False, "message": "vector_size is required when 'vectors' is not provided"}), 400

        create_collection(name, size, distance or "Cosine", vectors=vectors)
        return jsonify({"success": True, "message": f"Collection '{name}' created."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@qdrant_bp.route("/collection/list", methods=["GET"])
@authenticate
def list_collections_route():
    try:
        return jsonify({"success": True, "collections": get_all_collections()})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@qdrant_bp.route("/collection/info", methods=["GET"])
@authenticate
def collection_info_route():
    try:
        name = request.args.get("collection_name")
        if not name:
            return jsonify({"success": False, "message": "collection_name is required"}), 400
        return jsonify({"success": True, "info": get_collection_info(name)})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@qdrant_bp.route("/collection/delete", methods=["POST"])
@authenticate
def delete_collection_route():
    try:
        data = request.json or {}
        name = data.get("collection_name")
        if not name:
            return jsonify({"success": False, "message": "collection_name is required"}), 400
        delete_collection(name)
        return jsonify({"success": True, "message": f"Collection '{name}' deleted."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# -------- Vector endpoints --------
@qdrant_bp.route("/vector/upsert", methods=["POST"])
@authenticate
def upsert_vector_route():
    try:
        data = request.json or {}
        vector = data.get("vector")
        payload = data.get("payload", {})
        collection_name = data.get("collection_name", DEFAULT_COLLECTION)
        point_id = data.get("point_id")   # optional
        vector_name = data.get("vector_name")  # optional (named vectors)

        if vector is None:
            return jsonify({"success": False, "message": "vector is required"}), 400

        pid = save_vector(vector, payload, collection_name, point_id, vector_name=vector_name)
        return jsonify({"success": True, "point_id": pid})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@qdrant_bp.route("/vector/upsert-batch", methods=["POST"])
@authenticate
def upsert_batch_route():
    try:
        data = request.json or {}
        collection_name = data.get("collection_name", DEFAULT_COLLECTION)
        items = data.get("items")  # list of {point_id, vector, payload}
        vector_name = data.get("vector_name")
        if not isinstance(items, list) or not items:
            return jsonify({"success": False, "message": "items (list) is required"}), 400
        ids = save_vectors_batch(collection_name, items, vector_name=vector_name)
        return jsonify({"success": True, "point_ids": ids})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@qdrant_bp.route("/vector/search", methods=["POST"])
@authenticate
def search_vector_route():
    try:
        data = request.json or {}
        vector = data.get("vector")
        top_k = int(data.get("top_k", 3))
        collection_name = data.get("collection_name", DEFAULT_COLLECTION)
        include_vector = bool(data.get("include_vector", False))
        filters = data.get("filters", None)
        vector_name = data.get("vector_name")  # optional
        score_threshold = data.get("score_threshold")

        if vector is None:
            return jsonify({"success": False, "message": "vector is required"}), 400

        results = search_vector(
            vector, collection_name, top_k, include_vector, filters,
            vector_name=vector_name, score_threshold=score_threshold
        )
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@qdrant_bp.route("/vector/delete", methods=["POST"])
@authenticate
def delete_vector_route():
    try:
        data = request.json or {}
        point_ids = data.get("point_id")  # str | int | list
        collection_name = data.get("collection_name", DEFAULT_COLLECTION)
        if not point_ids:
            return jsonify({"success": False, "message": "point_id is required"}), 400
        delete_vector_by_id(point_ids, collection_name)
        return jsonify({"success": True, "message": f"Vector(s) {point_ids} deleted from '{collection_name}'"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ---- Tambahan ringan (sering dipakai), tetap dalam konteks ----
@qdrant_bp.route("/vector/get", methods=["POST"])
@authenticate
def get_vector_route():
    try:
        data = request.json or {}
        collection_name = data.get("collection_name", DEFAULT_COLLECTION)
        point_ids = data.get("point_id")
        include_vector = bool(data.get("include_vector", False))
        if not point_ids:
            return jsonify({"success": False, "message": "point_id is required"}), 400
        ids = point_ids if isinstance(point_ids, list) else [point_ids]
        res = get_vectors(collection_name, ids, include_vector=include_vector)
        return jsonify({"success": True, "results": res})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@qdrant_bp.route("/vector/set-payload", methods=["POST"])
@authenticate
def set_payload_route():
    try:
        data = request.json or {}
        collection_name = data.get("collection_name", DEFAULT_COLLECTION)
        point_ids = data.get("point_id")
        payload = data.get("payload")
        if not point_ids or not isinstance(payload, dict):
            return jsonify({"success": False, "message": "point_id and payload are required"}), 400
        ids = point_ids if isinstance(point_ids, list) else [point_ids]
        set_payload(collection_name, ids, payload)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@qdrant_bp.route("/vector/clear-payload", methods=["POST"])
@authenticate
def clear_payload_route():
    try:
        data = request.json or {}
        collection_name = data.get("collection_name", DEFAULT_COLLECTION)
        filters = data.get("filters")
        keys = data.get("keys")
        if not keys or not isinstance(keys, list):
            return jsonify({"success": False, "message": "keys (list) is required"}), 400
        clear_payload(collection_name, filters, keys)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@qdrant_bp.route("/vector/update-vector", methods=["POST"])
@authenticate
def update_vector_route():
    try:
        data = request.json or {}
        collection_name = data.get("collection_name", DEFAULT_COLLECTION)
        point_id = data.get("point_id")
        vector = data.get("vector")
        vector_name = data.get("vector_name")
        if point_id is None or vector is None:
            return jsonify({"success": False, "message": "point_id and vector are required"}), 400
        update_vector(collection_name, point_id, vector, vector_name=vector_name)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
