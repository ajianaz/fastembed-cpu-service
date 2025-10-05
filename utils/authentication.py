# utils/authentication.py
from flask import request, jsonify
from functools import wraps
import os
from typing import Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")

# === Flags dinamis ===
AUTH_APIKEY_ENABLE: bool = _env_bool("AUTH_APIKEY_ENABLE", True)
AUTH_JWT_ENABLE: bool    = _env_bool("AUTH_JWT_ENABLE", True)

# === API Key config ===
API_KEYS = frozenset(k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip())
API_KEY_HEADER: str = os.getenv("API_KEY_HEADER", "X-API-Key")

# === JWT config (opsional) ===
JWT_SECRET: str = os.getenv("JWT_SECRET", "")
JWT_ALGO: str   = os.getenv("JWT_ALGO", "HS256")
JWT_AUDIENCE: Optional[str] = os.getenv("JWT_AUD", None)
JWT_ISSUER: Optional[str]   = os.getenv("JWT_ISS", None)
JWT_LEEWAY: int = int(os.getenv("JWT_LEEWAY", "0"))  # detik toleransi clock skew

def _ok(data=None, code=200):
    return jsonify({"success": True, **(data or {})}), code

def _unauth(msg="Unauthorized", code=401):
    return jsonify({"success": False, "message": msg}), code

def _extract_bearer_token() -> Optional[str]:
    """Ambil token dari Authorization: Bearer <token> (jika ada)."""
    auth = request.headers.get("Authorization", "")
    if isinstance(auth, str) and auth.startswith("Bearer "):
        return auth[len("Bearer "):].strip()
    return None

def _check_apikey_enabled() -> bool:
    """Validasi API Key jika di-enable."""
    # 1) Header X-API-Key (utama untuk gateway)
    key = request.headers.get(API_KEY_HEADER)
    if key and key in API_KEYS:
        return True
    # 2) Authorization: Bearer <key> (fallback)
    token = _extract_bearer_token()
    if token and token in API_KEYS:
        return True
    return False

def _check_jwt_enabled() -> Tuple[bool, str]:
    """Validasi JWT jika di-enable. Return (ok, err_msg)."""
    token = _extract_bearer_token()
    if not token:
        return False, "Missing Bearer token"

    # Pastikan PyJWT ada
    try:
        import jwt  # PyJWT
    except Exception:
        return False, "JWT auth enabled but PyJWT not installed"

    # Pastikan secret tersedia
    if not JWT_SECRET:
        return False, "JWT_SECRET is not configured"

    decode_kwargs = {
        "algorithms": [JWT_ALGO],
        "leeway": JWT_LEEWAY,
        "options": {
            # Sesuaikan opsimu (misal bisa relax 'verify_aud' kalau tidak pakai audience)
            "require": [],  # contoh: ["exp", "iat"] bila perlu
        },
    }
    if JWT_AUDIENCE:
        decode_kwargs["audience"] = JWT_AUDIENCE
    if JWT_ISSUER:
        decode_kwargs["issuer"] = JWT_ISSUER

    try:
        # payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO], ...)
        payload = jwt.decode(token, JWT_SECRET, **decode_kwargs)  # noqa: F841
        # opsional: set g.user di sini
        return True, ""
    except Exception as e:
        return False, "Invalid token: {}".format(e)

def authenticate(f):
    """
    Kebijakan:
    - Jika kedua flag OFF → bypass (langsung lolos).
    - Jika salah satu ON:
        * API key ON  → valid jika header X-API-Key atau Bearer cocok daftar API_KEYS
        * JWT ON      → valid jika Authorization Bearer JWT valid (PyJWT)
      Jika keduanya ON, cukup salah satu valid → lolos.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        # Bypass total jika keduanya OFF (mis. ada gateway di depan)
        if not AUTH_APIKEY_ENABLE and not AUTH_JWT_ENABLE:
            return f(*args, **kwargs)

        # Bypass preflight CORS agar FE nggak ke-block
        if request.method == "OPTIONS":
            return _ok()

        apikey_ok = False
        jwt_ok = False
        jwt_err = ""

        if AUTH_APIKEY_ENABLE:
            apikey_ok = _check_apikey_enabled()

        if AUTH_JWT_ENABLE:
            jwt_ok, jwt_err = _check_jwt_enabled()

        # Jika keduanya ON → cukup salah satu lolos
        if AUTH_APIKEY_ENABLE and AUTH_JWT_ENABLE:
            if apikey_ok or jwt_ok:
                return f(*args, **kwargs)
            return _unauth(jwt_err or "Invalid API key")

        # Jika hanya API key ON
        if AUTH_APIKEY_ENABLE and not AUTH_JWT_ENABLE:
            return f(*args, **kwargs) if apikey_ok else _unauth("Invalid API key")

        # Jika hanya JWT ON
        if AUTH_JWT_ENABLE and not AUTH_APIKEY_ENABLE:
            return f(*args, **kwargs) if jwt_ok else _unauth(jwt_err or "Invalid token")

        # Default (harusnya tidak sampai sini)
        return _unauth()
    return decorated
