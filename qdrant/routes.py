from flask import Blueprint, request, jsonify
from .utils import (
    create_collection, get_all_collections, get_collection_info, delete_collection
)
from utils.authentication import authenticate
from .config import DEFAULT_COLLECTION


# You can change this name
qdrant_bp = Blueprint("qdrant", __name__)

@qdrant_bp.route("/collection/create", methods=["POST"])
@authenticate
def create_collection_route():
    try:
        data = request.json
        name = data.get("collection_name")
        size = data.get("vector_size")
        if not name or not size:
            return jsonify({"success": False, "message": "collection_name and vector_size are required"}), 400
        create_collection(name, size)
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
        name = request.json.get("collection_name")
        if not name:
            return jsonify({"success": False, "message": "collection_name is required"}), 400
        delete_collection(name)
        return jsonify({"success": True, "message": f"Collection '{name}' deleted."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

from .utils import (
    save_vector, search_vector, delete_vector_by_id
)

@qdrant_bp.route("/vector/upsert", methods=["POST"])
@authenticate
def upsert_vector_route():
    try:
        data = request.json
        vector = data.get("vector")
        payload = data.get("payload", {})
        collection_name = data.get("collection_name", DEFAULT_COLLECTION)
        point_id = data.get("point_id")  # Optional

        if not vector:
            return jsonify({"success": False, "message": "Vector is required"}), 400

        vector_id = save_vector(vector, payload, collection_name, point_id)
        return jsonify({"success": True, "point_id": vector_id})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@qdrant_bp.route("/vector/search", methods=["POST"])
@authenticate
def search_vector_route():
    try:
        data = request.json
        vector = data.get("vector")
        top_k = int(data.get("top_k", 3))
        collection_name = data.get("collection_name", DEFAULT_COLLECTION)
        include_vector = data.get("include_vector", False)
        filters = data.get("filters", None)

        if not vector:
            return jsonify({"success": False, "message": "Vector is required"}), 400

        results = search_vector(vector, collection_name, top_k, include_vector, filters)
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@qdrant_bp.route("/vector/delete", methods=["POST"])
@authenticate
def delete_vector_route():
    try:
        data = request.json
        point_ids = data.get("point_id")  # Bisa str atau list
        collection_name = data.get("collection_name", DEFAULT_COLLECTION)

        if not point_ids:
            return jsonify({"success": False, "message": "point_id is required"}), 400

        delete_vector_by_id(point_ids, collection_name)
        return jsonify({"success": True, "message": f"Vector(s) {point_ids} deleted from '{collection_name}'"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

