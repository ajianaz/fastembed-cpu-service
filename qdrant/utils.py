# qdrant/utils.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union

from qdrant_client.models import (
    Distance, VectorParams,
    Filter, FieldCondition, MatchValue, MatchAny, Range,
    PointStruct, NamedVector, ScoredPoint,
)

from .client import ensure_client
from .config import DEFAULT_DISTANCE

# -------- Distance helper --------
def _distance_from_str(name: str) -> Distance:
    low = (name or DEFAULT_DISTANCE or "Cosine").lower()
    if low in ("dot", "dotproduct"):
        return Distance.DOT
    if low in ("euclid", "l2", "euclidean"):
        return Distance.EUCLID
    return Distance.COSINE

# -------- Collection ops --------
def create_collection(
    name: str,
    vector_size: Optional[int] = None,
    distance: str = DEFAULT_DISTANCE,
    *,
    vectors: Optional[Dict[str, Dict[str, Union[int, str]]]] = None,
) -> None:
    cl = ensure_client()
    if cl is None:
        raise RuntimeError("Qdrant client is not configured")

    if vectors:
        cfg = {
            name_: VectorParams(size=int(v["size"]), distance=_distance_from_str(str(v.get("distance", distance))))
            for name_, v in vectors.items()
        }
        cl.recreate_collection(collection_name=name, vectors_config=cfg)
    else:
        if not vector_size:
            raise ValueError("vector_size is required when 'vectors' is not provided")
        cl.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=int(vector_size), distance=_distance_from_str(distance)),
        )

def get_all_collections() -> List[str]:
    cl = ensure_client()
    if cl is None:
        return []
    res = cl.get_collections()
    return [c.name for c in getattr(res, "collections", [])]

def get_collection_info(name: str) -> Dict[str, Any]:
    cl = ensure_client()
    if cl is None:
        raise RuntimeError("Qdrant client is not configured")
    info = cl.get_collection(name)
    return {
        "status": getattr(info, "status", None),
        "vectors_count": getattr(info, "vectors_count", None),
        "config": getattr(info, "config", None).dict() if getattr(info, "config", None) else None,
    }

def delete_collection(name: str) -> None:
    cl = ensure_client()
    if cl is None:
        raise RuntimeError("Qdrant client is not configured")
    cl.delete_collection(name)

# -------- Filter builder --------
def _cond_from_dict(d: Dict[str, Any]) -> FieldCondition:
    key = d.get("key")
    if not key:
        raise ValueError("filter condition requires 'key'")
    if "match" in d:
        v = d["match"].get("value")
        return FieldCondition(key=key, match=MatchValue(value=v))
    if "in" in d:
        vs = d["in"].get("values") or []
        return FieldCondition(key=key, match=MatchAny(any=vs))
    if "range" in d:
        r = d["range"]
        return FieldCondition(key=key, range=Range(
            gte=r.get("gte"), gt=r.get("gt"),
            lte=r.get("lte"), lt=r.get("lt"),
        ))
    raise ValueError("unsupported filter condition")

def build_filter(dsl: Optional[Dict[str, Any]]) -> Optional[Filter]:
    if not dsl:
        return None
    must = [ _cond_from_dict(c) for c in dsl.get("must", []) ]
    must_not = [ _cond_from_dict(c) for c in dsl.get("must_not", []) ]
    should = [ _cond_from_dict(c) for c in dsl.get("should", []) ]
    return Filter(must=must or None, must_not=must_not or None, should=should or None)

# -------- Vector ops --------
def save_vector(
    vector: List[float],
    payload: Dict[str, Any],
    collection_name: str,
    point_id: Optional[Union[str, int]] = None,
    *,
    vector_name: Optional[str] = None,
) -> Union[str, int]:
    cl = ensure_client()
    if cl is None:
        raise RuntimeError("Qdrant client is not configured")

    vec = NamedVector(name=vector_name, vector=vector) if vector_name else vector
    p = PointStruct(id=point_id, vector=vec, payload=payload)
    cl.upsert(collection_name=collection_name, points=[p])
    return p.id

def save_vectors_batch(
    collection_name: str,
    items: List[Dict[str, Any]],
    *,
    vector_name: Optional[str] = None,
) -> List[Union[str, int]]:
    cl = ensure_client()
    if cl is None:
        raise RuntimeError("Qdrant client is not configured")

    points = []
    for it in items:
        vec = it.get("vector")
        pid = it.get("point_id")
        payload = it.get("payload") or {}
        v = NamedVector(name=vector_name, vector=vec) if vector_name else vec
        points.append(PointStruct(id=pid, vector=v, payload=payload))
    cl.upsert(collection_name=collection_name, points=points)
    return [p.id for p in points]

def search_vector(
    vector: List[float],
    collection_name: str,
    top_k: int = 3,
    include_vector: bool = False,
    filters: Optional[Dict[str, Any]] = None,
    *,
    vector_name: Optional[str] = None,
    score_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    cl = ensure_client()
    if cl is None:
        raise RuntimeError("Qdrant client is not configured")

    qfilter = build_filter(filters)
    qvec = NamedVector(name=vector_name, vector=vector) if vector_name else vector

    # Kompatibel lintas versi: with_vectors bisa bool atau list nama vectors
    with_vectors_arg: Union[bool, List[str]] = False
    if include_vector:
        with_vectors_arg = [vector_name] if vector_name else True

    res: List[ScoredPoint] = cl.search(
        collection_name=collection_name,
        query_vector=qvec,
        limit=int(top_k),
        query_filter=qfilter,
        score_threshold=score_threshold,
        with_payload=True,
        with_vectors=with_vectors_arg,
    )
    out: List[Dict[str, Any]] = []
    for r in res:
        item = {"id": r.id, "score": r.score, "payload": r.payload}
        # r.vector tersedia jika with_vectors=True/list
        if include_vector and hasattr(r, "vector"):
            item["vector"] = r.vector
        out.append(item)
    return out

def delete_vector_by_id(
    point_ids: Union[Union[str, int], List[Union[str, int]]],
    collection_name: str
) -> None:
    cl = ensure_client()
    if cl is None:
        raise RuntimeError("Qdrant client is not configured")
    ids = point_ids if isinstance(point_ids, list) else [point_ids]
    cl.delete(collection_name=collection_name, points_selector=ids)

def get_vectors(
    collection_name: str,
    point_ids: List[Union[str, int]],
    include_vector: bool = False,
    *,
    vector_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    cl = ensure_client()
    # Kompatibel lintas versi
    with_vectors_arg: Union[bool, List[str]] = False
    if include_vector:
        with_vectors_arg = [vector_name] if vector_name else True

    res = cl.retrieve(
        collection_name=collection_name,
        ids=point_ids,
        with_payload=True,
        with_vectors=with_vectors_arg,
    )
    out = []
    for p in res:
        item = {"id": p.id, "payload": p.payload}
        if include_vector and hasattr(p, "vector"):
            item["vector"] = p.vector
        out.append(item)
    return out

def set_payload(
    collection_name: str,
    point_ids: List[Union[str, int]],
    payload: Dict[str, Any]
) -> None:
    cl = ensure_client()
    cl.set_payload(collection_name=collection_name, payload=payload, points=point_ids)

def clear_payload(
    collection_name: str,
    filters: Optional[Dict[str, Any]],
    keys: List[str]
) -> None:
    cl = ensure_client()
    qfilter = build_filter(filters)
    cl.delete_payload(collection_name=collection_name, keys=keys, wait=True, filter=qfilter)

def update_vector(
    collection_name: str,
    point_id: Union[str, int],
    vector: List[float],
    *,
    vector_name: Optional[str] = None
) -> None:
    cl = ensure_client()
    v = NamedVector(name=vector_name, vector=vector) if vector_name else vector
    cl.update_vectors(collection_name=collection_name, points=[(point_id, v)], wait=True)
