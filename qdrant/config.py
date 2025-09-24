"""
qdrant/config.py
Konfigurasi khusus modul Qdrant.
- Mendukung Qdrant Cloud (pakai QDRANT_URL) & lokal (pakai QDRANT_HOST/PORT).
- DEFAULT_COLLECTION dipakai saat route tidak diberi collection_name.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Dinamis: gunakan URL (Cloud) ATAU host/port (lokal)
QDRANT_ENABLE = os.getenv("QDRANT_ENABLE", "true").lower() == "true"
QDRANT_URL = (os.getenv("QDRANT_URL") or "").strip()  # contoh: https://xxxx.cloud.qdrant.io
QDRANT_HOST = (os.getenv("QDRANT_HOST") or "").strip()  # contoh: localhost / qdrant (tanpa http://)
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
QDRANT_API_KEY = (os.getenv("QDRANT_API_KEY") or "").strip() or None
PREFER_GRPC = os.getenv("PREFER_GRPC", "false").lower() == "true"

# Koleksi default jika client tidak menyebutkan nama koleksi
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "qdrant_default")

# Default distance (Cosine/Euclid/Dot)
DEFAULT_DISTANCE = os.getenv("QDRANT_DISTANCE", "Cosine")

# Timeout opsional (detik)
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "30"))
