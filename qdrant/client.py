import logging
from qdrant_client import QdrantClient
from .config import QDRANT_ENABLE, QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY, PREFER_GRPC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

qdrant_client = None

if QDRANT_ENABLE:
    try:
        qdrant_url = QDRANT_HOST if QDRANT_HOST.startswith("http") else f"http://{QDRANT_HOST}:{QDRANT_PORT}"
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=QDRANT_API_KEY or None,
            prefer_grpc=PREFER_GRPC,
        )
        qdrant_client.get_collections()
        logger.info(f"✅ Connected to Qdrant at {qdrant_url}")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant: {e}")
        qdrant_client = None
