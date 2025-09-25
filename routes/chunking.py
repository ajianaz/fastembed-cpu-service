# routes/chunking.py
from __future__ import annotations
from flask import Blueprint, request, jsonify

# Auth opsional
try:
    from utils.authentication import authenticate  # type: ignore
except Exception:
    def authenticate(fn):
        return fn

from utils.chunking import count_tokens
from utils.chunking import count_tokens_batch
from utils.chunking import auto_chunk_texts
from utils.chunking import attach_chunk_metadata
from utils.chunking import chunk_text as chunk_dispatch

chunk_bp = Blueprint("chunk_tools", __name__)

@chunk_bp.route("/tokens/count", methods=["POST"])
@authenticate
def tokens_count_route():
    """
    Body:
      { "inputs": ["a","b"], "model": "cl100k_base", "sum": true }
      atau
      { "text": "single text", "model": "cl100k_base" }
    """
    try:
        data = request.get_json(force=True, silent=False) or {}
        model = data.get("model")
        if "inputs" in data:
            inputs = data.get("inputs") or []
            if not isinstance(inputs, list):
                return jsonify({"success": False, "message": "inputs harus list"}), 400
            tokens = count_tokens_batch(inputs, model)
            resp = {"success": True, "tokens": tokens}
            if data.get("sum", False):
                resp["total"] = sum(tokens)
            return jsonify(resp)
        text = data.get("text") or ""
        return jsonify({"success": True, "tokens": count_tokens(text, model)})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@chunk_bp.route("/chunk/text", methods=["POST"])
@authenticate
def chunk_text_route():
    """
    Body:
    {
      "text": "...",
      "mode": "tokens|chars|paragraph|sentence|lines|regex|markdown",
      "enable_auto_chunk": true,
      "max_tokens": 800, "overlap_tokens": 80, "model": "cl100k_base",
      "max_chars": 2000, "overlap_chars": 200,
      "max_lines": 40, "overlap_lines": 5,
      "pattern": "\\n##\\s", "keep_delimiter": true, "regex_max_chars": 4000,
      "md_min_level": 2,
      "metadata": { ... }, "source_id": "..."
    }
    """
    try:
        data = request.get_json(force=True, silent=False) or {}
        text = data.get("text", "")
        mode = (data.get("mode") or "tokens").lower()
        base_meta = data.get("metadata") or {}
        source_id = data.get("source_id")

        if mode == "tokens" and data.get("enable_auto_chunk", True):
            max_tokens = int(data.get("max_tokens", 800))
            overlap_tokens = int(data.get("overlap_tokens", 80))
            model = data.get("model")
            items, stats = auto_chunk_texts(
                [text],
                enable_auto_chunk=True,
                model_name=model,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
            )
            chunks = attach_chunk_metadata(base_meta, items[0]["chunks"], source_id=source_id)
            return jsonify({"success": True, "stats": stats, "chunks": chunks})

        chunks_raw = chunk_dispatch(
            text,
            mode=mode,
            max_tokens=int(data.get("max_tokens", 800)),
            overlap_tokens=int(data.get("overlap_tokens", 80)),
            model_name=data.get("model"),
            max_chars=int(data.get("max_chars", 2000)),
            overlap_chars=int(data.get("overlap_chars", 200)),
            max_lines=int(data.get("max_lines", 40)),
            overlap_lines=int(data.get("overlap_lines", 5)),
            joiner=data.get("joiner", "\n\n"),
            pattern=data.get("pattern"),
            keep_delimiter=bool(data.get("keep_delimiter", True)),
            regex_max_chars=data.get("regex_max_chars"),
            md_min_level=int(data.get("md_min_level", 2)),
        )
        chunks = attach_chunk_metadata(base_meta, chunks_raw, source_id=source_id)
        stats = {"mode": mode, "chunks": len(chunks)}
        return jsonify({"success": True, "stats": stats, "chunks": chunks})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@chunk_bp.route("/chunk/batch", methods=["POST"])
@authenticate
def chunk_batch_route():
    """
    Body:
    {
      "inputs": [
        {"text":"...","mode":"sentence","metadata":{...},"source_id":"..."},
        {"text":"...","mode":"paragraph","metadata":{...}}
      ],
      "enable_auto_chunk": true,
      "max_tokens": 800, "overlap_tokens": 80, "model": "cl100k_base",
      "max_chars": 2000, "overlap_chars": 200,
      "max_lines": 40, "overlap_lines": 5,
      "pattern": "\\n##\\s", "keep_delimiter": true, "regex_max_chars": 4000,
      "md_min_level": 2
    }
    """
    try:
        data = request.get_json(force=True, silent=False) or {}
        inputs = data.get("inputs") or []
        if not isinstance(inputs, list):
            return jsonify({"success": False, "message": "inputs harus list"}), 400

        g_enable_auto = bool(data.get("enable_auto_chunk", True))
        g_max_tokens = int(data.get("max_tokens", 800))
        g_overlap_tokens = int(data.get("overlap_tokens", 80))
        g_model = data.get("model")
        g_max_chars = int(data.get("max_chars", 2000))
        g_overlap_chars = int(data.get("overlap_chars", 200))
        g_max_lines = int(data.get("max_lines", 40))
        g_overlap_lines = int(data.get("overlap_lines", 5))
        g_joiner = data.get("joiner", "\n\n")
        g_pattern = data.get("pattern")
        g_keep_delimiter = bool(data.get("keep_delimiter", True))
        g_regex_max_chars = data.get("regex_max_chars")
        g_md_min_level = int(data.get("md_min_level", 2))

        items_out = []
        total_chunks = 0

        for idx, it in enumerate(inputs):
            text = it.get("text", "")
            mode = (it.get("mode") or "tokens").lower()
            base_meta = it.get("metadata") or {}
            source_id = it.get("source_id")

            enable_auto = bool(it.get("enable_auto_chunk", g_enable_auto))
            max_tokens = int(it.get("max_tokens", g_max_tokens))
            overlap_tokens = int(it.get("overlap_tokens", g_overlap_tokens))
            model = it.get("model", g_model)
            max_chars = int(it.get("max_chars", g_max_chars))
            overlap_chars = int(it.get("overlap_chars", g_overlap_chars))
            max_lines = int(it.get("max_lines", g_max_lines))
            overlap_lines = int(it.get("overlap_lines", g_overlap_lines))
            joiner = it.get("joiner", g_joiner)
            pattern = it.get("pattern", g_pattern)
            keep_delimiter = bool(it.get("keep_delimiter", g_keep_delimiter))
            regex_max_chars = it.get("regex_max_chars", g_regex_max_chars)
            md_min_level = int(it.get("md_min_level", g_md_min_level))

            if mode == "tokens" and enable_auto:
                arr, _stats = auto_chunk_texts(
                    [text],
                    enable_auto_chunk=True,
                    model_name=model,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
                chunks = attach_chunk_metadata(base_meta, arr[0]["chunks"], source_id=source_id)
            else:
                raw = chunk_dispatch(
                    text,
                    mode=mode,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    model_name=model,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars,
                    max_lines=max_lines,
                    overlap_lines=overlap_lines,
                    joiner=joiner,
                    pattern=pattern,
                    keep_delimiter=keep_delimiter,
                    regex_max_chars=regex_max_chars,
                    md_min_level=md_min_level,
                )
                chunks = attach_chunk_metadata(base_meta, raw, source_id=source_id)

            total_chunks += len(chunks)
            items_out.append({
                "source_index": idx,
                "chunks": chunks
            })

        stats = {"items": len(items_out), "total_chunks": total_chunks}
        return jsonify({"success": True, "stats": stats, "items": items_out})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
