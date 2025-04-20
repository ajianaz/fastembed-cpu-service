import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_ENABLE = os.getenv("QDRANT_ENABLE", "False").lower() == "true"
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "qdrant_default")
PREFER_GRPC = os.getenv("PREFER_GRPC", "False").lower() == "true"
