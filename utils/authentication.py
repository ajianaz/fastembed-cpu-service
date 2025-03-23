from flask import request, jsonify
from functools import wraps
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API keys from environment variable (comma-separated values)
API_KEYS = [key.strip() for key in os.getenv("API_KEYS", "adk_default").split(",")]
print("Loaded API_KEYS:", API_KEYS)

def authenticate(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get the Authorization header
        auth_header = request.headers.get("Authorization", "")

        # Extract the token (format: "Bearer <token>")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Unauthorized"}), 401

        token = auth_header[len("Bearer "):]

        # Check if the token exists in the allowed API keys
        if token not in API_KEYS:
            print(f"Unauthorized attempt with token: {token}")
            return jsonify({"error": "Unauthorized"}), 401

        return f(*args, **kwargs)
    return decorated_function
