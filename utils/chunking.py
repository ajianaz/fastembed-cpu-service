"""
utils/chunking.py
-----------------
Fungsi inti untuk:
- hitung token (single/batch)
- chunking berbasis token (dengan overlap)
- auto-chunk untuk list teks + guard limit via ENV
- berbagai opsi chunking: chars/paragraph/sentence/lines/regex/markdown
- semat metadata standar pada tiap chunk
- dispatcher chunk_text(mode=...)

Catatan:
- Tergantung tiktoken untuk mode tokens; pasang via `pip install tiktoken`.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os
import re

# ====== LIMIT via ENV ======
MAX_INPUT_CHARS_NO_CHUNK = int(os.getenv("MAX_INPUT_CHARS_NO_CHUNK", "8000"))
MAX_TOTAL_CHARS_PER_REQUEST = int(os.getenv("MAX_TOTAL_CHARS_PER_REQUEST", "200000"))
MAX_ITEMS_NO_CHUNK = int(os.getenv("MAX_ITEMS_NO_CHUNK", "128"))

# ====== tiktoken ======
try:
    import tiktoken
except Exception:
    tiktoken = None

def _get_encoding(model_name: Optional[str] = None):
    if tiktoken is None:
        raise RuntimeError("tiktoken belum terpasang. Jalankan: pip install tiktoken")
    if not model_name:
        return tiktoken.get_encoding("cl100k_base")
    aliases = {
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-4o": "o200k_base",
        "o200k_base": "o200k_base",
        "cl100k_base": "cl100k_base",
    }
    enc_name = aliases.get(model_name, "cl100k_base")
    return tiktoken.get_encoding(enc_name)

# ====== Token counting ======
def count_tokens(text: str, model_name: Optional[str] = None) -> int:
    enc = _get_encoding(model_name)
    return len(enc.encode(text or ""))

def count_tokens_batch(texts: List[str], model_name: Optional[str] = None) -> List[int]:
    enc = _get_encoding(model_name)
    return [len(enc.encode(t or "")) for t in texts]

# ====== Chunking by tokens ======
def chunk_text_by_tokens(
    text: str,
    max_tokens: int = 800,
    overlap_tokens: int = 80,
    model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens harus > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens tidak boleh negatif")

    enc = _get_encoding(model_name)
    tokens = enc.encode(text or "")
    n = len(tokens)
    if n == 0:
        return [{"text": "", "token_start": 0, "token_end": 0, "index": 0}]

    chunks: List[Dict[str, Any]] = []
    start = 0
    idx = 0
    step = max(1, max_tokens - overlap_tokens)

    while start < n:
        end = min(n, start + max_tokens)
        seg_toks = tokens[start:end]
        seg_text = enc.decode(seg_toks)
        chunks.append({
            "text": seg_text,
            "token_start": start,
            "token_end": end,
            "index": idx,
        })
        idx += 1
        start += step

    return chunks

# ====== Auto-chunk list teks + guard ======
def auto_chunk_texts(
    texts: List[str],
    *,
    enable_auto_chunk: bool = True,
    model_name: Optional[str] = None,
    max_tokens: int = 800,
    overlap_tokens: int = 80,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    total_chars = sum(len(t or "") for t in texts)
    if total_chars > MAX_TOTAL_CHARS_PER_REQUEST:
        raise ValueError(
            f"Total input chars ({total_chars}) > MAX_TOTAL_CHARS_PER_REQUEST={MAX_TOTAL_CHARS_PER_REQUEST}"
        )
    if not enable_auto_chunk and len(texts) > MAX_ITEMS_NO_CHUNK:
        raise ValueError(
            f"Jumlah item ({len(texts)}) > MAX_ITEMS_NO_CHUNK={MAX_ITEMS_NO_CHUNK} saat auto-chunk OFF"
        )

    enc = _get_encoding(model_name)
    items: List[Dict[str, Any]] = []
    total_chunks = 0

    for i, raw in enumerate(texts):
        s = raw or ""
        tok_count = len(enc.encode(s))
        entry: Dict[str, Any] = {
            "source_index": i,
            "original_chars": len(s),
            "original_tokens": tok_count,
            "chunks": [],
        }

        if enable_auto_chunk:
            entry["chunks"] = chunk_text_by_tokens(
                s, max_tokens=max_tokens, overlap_tokens=overlap_tokens, model_name=model_name
            )
        else:
            if len(s) > MAX_INPUT_CHARS_NO_CHUNK:
                raise ValueError(
                    f"Panjang item[{i}]={len(s)} chars > MAX_INPUT_CHARS_NO_CHUNK={MAX_INPUT_CHARS_NO_CHUNK}. "
                    f"Aktifkan auto-chunk untuk memproses dokumen panjang."
                )
            entry["chunks"] = [{
                "text": s,
                "token_start": 0,
                "token_end": tok_count,
                "index": 0
            }]

        total_chunks += len(entry["chunks"])
        items.append(entry)

    stats = {
        "total_input_items": len(texts),
        "total_input_chars": total_chars,
        "total_output_chunks": total_chunks,
        "auto_chunk": enable_auto_chunk,
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens,
        "model_name": model_name or "cl100k_base",
    }
    return items, stats

# ====== Metadata helper ======
def attach_chunk_metadata(
    base_metadata: Dict[str, Any],
    chunk_list: List[Dict[str, Any]],
    *,
    source_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    total = len(chunk_list)
    for c in chunk_list:
        meta = dict(base_metadata or {})
        meta.update({
            "chunk_index": c.get("index", 0),
            "chunk_count": total,
            "token_start": c.get("token_start", c.get("char_start", c.get("line_start", 0))),
            "token_end": c.get("token_end", c.get("char_end", c.get("line_end", 0))),
        })
        if source_id:
            meta["source_id"] = source_id
        out.append({"text": c.get("text", ""), "metadata": meta})
    return out

# ====== Chunking by characters ======
def chunk_text_by_chars(
    text: str,
    max_chars: int = 2000,
    overlap_chars: int = 200,
) -> List[Dict[str, Any]]:
    if max_chars <= 0:
        raise ValueError("max_chars harus > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars tidak boleh negatif")

    s = text or ""
    n = len(s)
    if n == 0:
        return [{"text": "", "char_start": 0, "char_end": 0, "index": 0}]

    chunks: List[Dict[str, Any]] = []
    start = 0
    idx = 0
    step = max(1, max_chars - overlap_chars)

    while start < n:
        end = min(n, start + max_chars)
        piece = s[start:end]
        chunks.append({
            "text": piece,
            "char_start": start,
            "char_end": end,
            "index": idx
        })
        idx += 1
        start += step

    return chunks

# ====== Paragraphs ======
_para_splitter = re.compile(r"\n\s*\n+")

def split_paragraphs(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    parts = _para_splitter.split(text)
    return [p.strip() for p in parts if p.strip() != ""]

def chunk_text_by_paragraphs(
    text: str,
    max_chars: int = 3000,
    joiner: str = "\n\n",
) -> List[Dict[str, Any]]:
    paras = split_paragraphs(text)
    chunks: List[Dict[str, Any]] = []
    buf: List[str] = []
    size = 0
    idx = 0

    for i, p in enumerate(paras):
        add_len = len(p) + (len(joiner) if buf else 0)
        if size + add_len > max_chars and buf:
            combined = joiner.join(buf)
            chunks.append({"text": combined, "para_start_index": i - len(buf), "para_end_index": i - 1, "index": idx})
            idx += 1
            buf = [p]
            size = len(p)
        else:
            buf.append(p)
            size += add_len

    if buf:
        combined = joiner.join(buf)
        start_i = len(paras) - len(buf)
        chunks.append({"text": combined, "para_start_index": start_i, "para_end_index": len(paras) - 1, "index": idx})

    if not chunks:
        chunks = [{"text": "", "para_start_index": 0, "para_end_index": 0, "index": 0}]
    return chunks

# ====== Sentences ======
_sentence_splitter = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]
    parts = _sentence_splitter.split(text)
    return [p.strip() for p in parts if p.strip() != ""]

def chunk_text_by_sentences(
    text: str,
    max_chars: int = 2000,
    joiner: str = " ",
) -> List[Dict[str, Any]]:
    sents = split_sentences(text)
    chunks: List[Dict[str, Any]] = []
    buf: List[str] = []
    size = 0
    idx = 0

    for i, s in enumerate(sents):
        add_len = len(s) + (len(joiner) if buf else 0)
        if size + add_len > max_chars and buf:
            combined = joiner.join(buf)
            chunks.append({"text": combined, "sent_start_index": i - len(buf), "sent_end_index": i - 1, "index": idx})
            idx += 1
            buf = [s]
            size = len(s)
        else:
            buf.append(s)
            size += add_len

    if buf:
        combined = joiner.join(buf)
        start_i = len(sents) - len(buf)
        chunks.append({"text": combined, "sent_start_index": start_i, "sent_end_index": len(sents) - 1, "index": idx})

    if not chunks:
        chunks = [{"text": "", "sent_start_index": 0, "sent_end_index": 0, "index": 0}]
    return chunks

# ====== Lines ======
def split_lines(text: str) -> List[str]:
    return (text or "").splitlines()

def chunk_text_by_lines(
    text: str,
    max_lines: int = 40,
    overlap_lines: int = 5,
    joiner: str = "\n",
) -> List[Dict[str, Any]]:
    if max_lines <= 0:
        raise ValueError("max_lines harus > 0")
    if overlap_lines < 0:
        raise ValueError("overlap_lines tidak boleh negatif")

    lines = split_lines(text)
    n = len(lines)
    if n == 0:
        return [{"text": "", "line_start": 0, "line_end": 0, "index": 0}]

    chunks: List[Dict[str, Any]] = []
    start = 0
    idx = 0
    step = max(1, max_lines - overlap_lines)

    while start < n:
        end = min(n, start + max_lines)
        piece = joiner.join(lines[start:end])
        chunks.append({"text": piece, "line_start": start, "line_end": end, "index": idx})
        idx += 1
        start += step

    return chunks

# ====== Regex ======
def chunk_text_by_regex(
    text: str,
    pattern: str,
    keep_delimiter: bool = True,
    max_chars: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not pattern:
        raise ValueError("pattern regex harus diisi")

    parts = re.split(f"({pattern})", text or "")
    tokens: List[str] = []
    for i in range(0, len(parts), 2):
        chunk = parts[i] or ""
        delim = parts[i + 1] if i + 1 < len(parts) else ""
        if keep_delimiter and delim:
            tokens.append(chunk + delim)
        else:
            if chunk:
                tokens.append(chunk)
            if delim:
                tokens.append(delim)

    if not max_chars:
        out = []
        for idx, t in enumerate(tokens):
            out.append({"text": t, "index": idx})
        return out

    out2: List[Dict[str, Any]] = []
    buf: List[str] = []
    size = 0
    idx = 0
    for t in tokens:
        add = len(t)
        if size + add > max_chars and buf:
            out2.append({"text": "".join(buf), "index": idx})
            idx += 1
            buf = [t]
            size = len(t)
        else:
            buf.append(t)
            size += add
    if buf:
        out2.append({"text": "".join(buf), "index": idx})
    if not out2:
        out2 = [{"text": "", "index": 0}]
    return out2

# ====== Markdown headings ======
_md_heading = re.compile(r"^(#{1,6})\s+(.*)$", flags=re.MULTILINE)

def chunk_text_by_markdown_headings(
    text: str,
    max_chars: int = 4000,
    min_level: int = 2,
) -> List[Dict[str, Any]]:
    s = text or ""
    if not s.strip():
        return [{"text": "", "index": 0}]

    matches = list(_md_heading.finditer(s))
    if not matches:
        return chunk_text_by_chars(s, max_chars=max_chars, overlap_chars=0)

    points: List[int] = [0, len(s)]
    for m in matches:
        level = len(m.group(1))
        if level >= min_level:
            points.append(m.start())

    points = sorted(set(points))
    sections: List[Tuple[int, int]] = []
    for i in range(len(points) - 1):
        sections.append((points[i], points[i + 1]))

    out: List[Dict[str, Any]] = []
    idx = 0
    for (a, b) in sections:
        segment = s[a:b]
        if len(segment) <= max_chars:
            out.append({"text": segment.strip(), "index": idx})
            idx += 1
        else:
            sub = chunk_text_by_chars(segment, max_chars=max_chars, overlap_chars=0)
            for ch in sub:
                out.append({"text": ch["text"].strip(), "index": idx})
                idx += 1

    if not out:
        out = [{"text": s.strip(), "index": 0}]
    return out

# ====== Dispatcher ======
def chunk_text(
    text: str,
    mode: str = "tokens",
    *,
    max_tokens: int = 800,
    overlap_tokens: int = 80,
    model_name: Optional[str] = None,
    max_chars: int = 2000,
    overlap_chars: int = 200,
    max_lines: int = 40,
    overlap_lines: int = 5,
    joiner: str = "\n\n",
    pattern: Optional[str] = None,
    keep_delimiter: bool = True,
    regex_max_chars: Optional[int] = None,
    md_min_level: int = 2,
) -> List[Dict[str, Any]]:
    mode = (mode or "tokens").lower()
    if mode == "tokens":
        return chunk_text_by_tokens(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens, model_name=model_name)
    if mode == "chars":
        return chunk_text_by_chars(text, max_chars=max_chars, overlap_chars=overlap_chars)
    if mode == "paragraph":
        return chunk_text_by_paragraphs(text, max_chars=max_chars, joiner=joiner)
    if mode == "sentence":
        return chunk_text_by_sentences(text, max_chars=max_chars, joiner=" ")
    if mode == "lines":
        return chunk_text_by_lines(text, max_lines=max_lines, overlap_lines=overlap_lines, joiner="\n")
    if mode == "regex":
        if not pattern:
            raise ValueError("mode=regex membutuhkan 'pattern'")
        return chunk_text_by_regex(text, pattern=pattern, keep_delimiter=keep_delimiter, max_chars=regex_max_chars)
    if mode == "markdown":
        return chunk_text_by_markdown_headings(text, max_chars=max_chars, min_level=md_min_level)
    raise ValueError(f"mode tidak dikenal: {mode}")

__all__ = [
    "count_tokens",
    "count_tokens_batch",
    "chunk_text_by_tokens",
    "auto_chunk_texts",
    "attach_chunk_metadata",
    "chunk_text_by_chars",
    "chunk_text_by_paragraphs",
    "chunk_text_by_sentences",
    "chunk_text_by_lines",
    "chunk_text_by_regex",
    "chunk_text_by_markdown_headings",
    "chunk_text",
]