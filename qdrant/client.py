"""
qdrant/client.py
Helper inisialisasi QdrantClient yang fleksibel:
- Cloud: gunakan QDRANT_URL (HTTPS) + API key
- Lokal/Docker: gunakan host/port (tanpa http://), bisa prefer gRPC
"""

from __future__ import annotations
import os
from typing import Optional
from qdrant_client import QdrantClient
from .config import (
    QDRANT_ENABLE, QDRANT_URL, QDRANT_HOST, QDRANT_PORT, QDRANT_GRPC_PORT,
    QDRANT_API_KEY, PREFER_GRPC, QDRANT_TIMEOUT
)

_client: Optional[QdrantClient] = None

def ensure_client() -> Optional[QdrantClient]:
    global _client
    if _client is not None:
        return _client
    if not QDRANT_ENABLE:
        return None

    # 1) Cloud / URL penuh
    if QDRANT_URL:
        _client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=QDRANT_TIMEOUT,
        )
        return _client

    # 2) Lokal / host:port
    host = (QDRANT_HOST or "").strip()
    if not host:
        raise RuntimeError("Qdrant enabled but neither QDRANT_URL nor QDRANT_HOST is set.")

    # Jika user menaruh http:// di HOST, fallback pakai url=
    if host.startswith("http://") or host.startswith("https://"):
        url = f"{host}:{QDRANT_PORT}"
        _client = QdrantClient(url=url, api_key=QDRANT_API_KEY, timeout=QDRANT_TIMEOUT, prefer_grpc=False)
        return _client

    # Host “bersih”: boleh pilih HTTP/gRPC via prefer_grpc
    _client = QdrantClient(
        host=host,
        port=QDRANT_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        api_key=QDRANT_API_KEY,
        prefer_grpc=PREFER_GRPC,
        timeout=QDRANT_TIMEOUT,
    )
    return _client
