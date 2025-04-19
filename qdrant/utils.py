import uuid
import logging
from typing import Union, List, Dict, Any, Optional
from qdrant_client.http.models import (
    Distance, VectorParams, FieldCondition, MatchValue,
    Filter, SearchParams, PointIdsList
)
from qdrant_client.http.exceptions import UnexpectedResponse

from .client import qdrant_client
from .config import DEFAULT_COLLECTION

logger = logging.getLogger(__name__)

# === Collection Handling ===
def ensure_collection(collection_name: str, vector_size: int, distance=Distance.COSINE):
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized.")
    try:
        qdrant_client.get_collection(collection_name)
    except UnexpectedResponse:
        logger.info(f"🔧 Creating collection '{collection_name}'...")
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )

# === Save Vector ===
def save_vector(vector: List[float], payload: Dict[str, Any],
                collection_name: str = DEFAULT_COLLECTION,
                point_id: Optional[str] = None) -> str:
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized.")
    ensure_collection(collection_name, len(vector))
    point_id = point_id or str(uuid.uuid4())
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[{"id": point_id, "vector": vector, "payload": payload}]
    )
    logger.info(f"✅ Saved vector ID {point_id} to '{collection_name}'")
    return point_id

# === Delete Vector ===
def delete_vector_by_id(point_ids: Union[str, List[str]], collection_name: str = DEFAULT_COLLECTION):
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized.")

    # Normalize to list
    if isinstance(point_ids, str):
        point_ids = [point_ids]

    qdrant_client.delete(
        collection_name=collection_name,
        points_selector=PointIdsList(points=point_ids)
    )
    logger.info(f"🗑️ Deleted vector(s) ID {point_ids} from '{collection_name}'")


# === Filter Builder ===
def build_filter(dynamic_filter: Dict[str, Any]) -> Optional[Filter]:
    if not dynamic_filter:
        return None

    must, should, must_not = [], [], []

    # Handle 'must' filters dynamically
    for condition in dynamic_filter.get("must", []):
        if isinstance(condition, dict):
            for key, value in condition.items():
                if isinstance(value, dict):
                    for range_operator, range_value in value.items():
                        # Directly use FieldCondition with range operator
                        if range_operator == "gte":
                            must.append(FieldCondition(key=key, range={"gte": range_value}))
                        elif range_operator == "lte":
                            must.append(FieldCondition(key=key, range={"lte": range_value}))
                        elif range_operator == "gt":
                            must.append(FieldCondition(key=key, range={"gt": range_value}))
                        elif range_operator == "lt":
                            must.append(FieldCondition(key=key, range={"lt": range_value}))
                else:
                    must.append(FieldCondition(key=key, match=MatchValue(value=value)))

    return Filter(
        must=must if must else None,
        should=should if should else None,
        must_not=must_not if must_not else None
    )




# === Search Vector ===
def search_vector(
    vector: List[float],
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 3,
    include_vector: bool = False,
    filters: Optional[Dict[str, Any]] = None,
    search_params: Optional[SearchParams] = None
) -> List[Dict[str, Any]]:
    if not qdrant_client:
        raise RuntimeError("Qdrant client not initialized.")

    # Build the filter object using the provided dynamic filter
    filter_obj = build_filter(filters)

    # Perform the search query in Qdrant
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=vector,
        limit=top_k,
        query_filter=filter_obj,
        with_payload=True,  # Ensure payload is included in the result
        search_params=search_params
    )

    # Format and return the results
    return [{
        "id": str(r.id),
        "score": r.score,
        "payload": r.payload,
        **({"vector": r.vector} if include_vector else {})  # Include vector if requested
    } for r in results]



# === Collection Management ===
def create_collection(collection_name: str, vector_size: int, distance=Distance.COSINE):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=distance),
    )
    logger.info(f"📦 Created collection '{collection_name}'")

def get_all_collections() -> List[str]:
    collections = qdrant_client.get_collections()
    return [c.name for c in collections.collections]

def get_collection_info(collection_name: str) -> Dict[str, Any]:
    return qdrant_client.get_collection(collection_name).dict()

def delete_collection(collection_name: str):
    qdrant_client.delete_collection(collection_name=collection_name)
    logger.info(f"🗑️ Deleted collection '{collection_name}'")
