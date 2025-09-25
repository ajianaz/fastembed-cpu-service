# utils/preprocess.py
from __future__ import annotations
import re
import unicodedata
from typing import Dict

_CTRL = "".join(chr(c) for c in range(0x00, 0x20) if chr(c) not in ("\n", "\t"))
_CTRL_RE = re.compile(f"[{re.escape(_CTRL)}]")
_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

def clean_text(
    text: str,
    options: Dict = None
) -> str:
    """
    Bersihkan teks sesuai opsi.
    options:
      - normalize_unicode: bool = True
      - collapse_whitespace: bool = True
      - collapse_newlines: bool = False     # \n\n\n -> \n\n
      - strip_control_chars: bool = True
      - remove_urls: bool = False
      - lowercase: bool = False
      - strip_headers_footers: bool = False # hapus pola header/footer yg repetitif sederhana
      - header_footer_hint: str | None      # pola teks yang sering muncul (opsional)

    Catatan: default aman & non-destruktif terhadap makna.
    """
    if options is None:
        options = {}

    s = text or ""

    if options.get("normalize_unicode", True):
        s = unicodedata.normalize("NFKC", s)

    if options.get("strip_control_chars", True):
        s = _CTRL_RE.sub("", s)

    if options.get("remove_urls", False):
        s = _URL_RE.sub("", s)

    if options.get("collapse_whitespace", True):
        # rapikan spasi/tab; bukan newline
        s = _WS_RE.sub(" ", s)
        # strip per baris
        s = "\n".join(line.strip() for line in s.splitlines())

    if options.get("collapse_newlines", False):
        # batasi newline berturut-turut jadi maksimal dua
        s = _MULTI_NL_RE.sub("\n\n", s)

    if options.get("strip_headers_footers", False):
        # strategi sederhana: jika ada hint, drop baris yang mengandung hint
        hint = options.get("header_footer_hint")
        if hint:
            lines = s.splitlines()
            keep = [ln for ln in lines if hint not in ln]
            s = "\n".join(keep)
        # tambahan: hapus nomor halaman tunggal di baris sendiri, mis. "Page 12" / "12"
        s = re.sub(r"(?m)^(?:page\s*\d+|\d+)\s*$", "", s, flags=re.IGNORECASE)

    if options.get("lowercase", False):
        s = s.lower()

    # final trim
    return s.strip()
