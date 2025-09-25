# app.py
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# --- Load ENV ---
load_dotenv()

# --- Blueprints ---
from routes.embeddings import embeddings_bp           # asumsi sudah ada
from routes.chunking import chunk_bp                  # dari file chunking routes
from qdrant.routes import qdrant_bp                   # CRUD Qdrant

def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, supports_credentials=True)

    # Healthcheck
    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    # Register blueprints (pakai prefix seragam /api; ubah kalau mau tanpa prefix)
    app.register_blueprint(embeddings_bp)
    app.register_blueprint(chunk_bp)
    app.register_blueprint(qdrant_bp)

    return app

# Entrypoint
if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000)
else:
    app = create_app()
