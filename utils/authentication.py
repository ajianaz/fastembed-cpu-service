# utils/authentication.py (atau file auth kamu saat ini)
from flask import request, jsonify
from functools import wraps
import os
from dotenv import load_dotenv

load_dotenv()

# === Flags dinamis ===
AUTH_APIKEY_ENABLE = os.getenv("AUTH_APIKEY_ENABLE", "true").lower() == "true"
AUTH_JWT_ENABLE    = os.getenv("AUTH_JWT_ENABLE", "true").lower() == "true"

# === API Key config ===
API_KEYS = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "X-API-Key")

# === JWT config (opsional) ===
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGO   = os.getenv("JWT_ALGO", "HS256")

def _ok(data=None, code=200):
    return jsonify({"success": True, **(data or {})}), code

def _unauth(msg="Unauthorized", code=401):
    return jsonify({"success": False, "message": msg}), code

def _extract_bearer_token() -> str | None:
    """Ambil token dari Authorization: Bearer <token> (jika ada)."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
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

def _check_jwt_enabled() -> tuple[bool, str]:
    """Validasi JWT jika di-enable. Return (ok, err_msg)."""
    token = _extract_bearer_token()
    if not token:
        return False, "Missing Bearer token"
    try:
        import jwt  # PyJWT
    except Exception:
        return False, "JWT auth enabled but PyJWT not installed"

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        # optional: bisa set request context user di sini kalau perlu
        # g.user = {"sub": payload.get("sub"), "roles": payload.get("roles", [])}
        return True, ""
    except Exception as e:
        return False, f"Invalid token: {e}"

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

        apikey_ok = False
        jwt_ok = False

        if AUTH_APIKEY_ENABLE:
            apikey_ok = _check_apikey_enabled()

        if AUTH_JWT_ENABLE:
            jwt_ok, jwt_err = _check_jwt_enabled()
        else:
            jwt_err = ""

        # Jika keduanya ON → cukup salah satu lolos
        # Jika hanya satu ON → yang itu harus lolos
        if AUTH_APIKEY_ENABLE and AUTH_JWT_ENABLE:
            if apikey_ok or jwt_ok:
                return f(*args, **kwargs)
            # Prioritaskan pesan error yang paling relevan
            return _unauth(jwt_err or "Invalid API key")

        if AUTH_APIKEY_ENABLE and not AUTH_JWT_ENABLE:
            return f(*args, **kwargs) if apikey_ok else _unauth("Invalid API key")

        if AUTH_JWT_ENABLE and not AUTH_APIKEY_ENABLE:
            return f(*args, **kwargs) if jwt_ok else _unauth(jwt_err or "Invalid token")

        # Default (shouldn't reach)
        return _unauth()

    return decorated
