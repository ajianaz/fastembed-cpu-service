from flask import Blueprint, request, jsonify
from utils.utils import calculate_token_count
from utils.authentication import authenticate

# Create a blueprint
token_calculation_bp = Blueprint("token_calculation", __name__)

@token_calculation_bp.route("/v1/calculate-tokens", methods=["POST"])
@authenticate
def calculate_tokens_endpoint():
    """
    Endpoint to calculate tokens for given input text.
    """
    data = request.get_json()

    if not data or "input" not in data:
        return jsonify({"error": "Missing 'input' field"}), 400

    input_text = data.get("input", "")
    model = data.get("model", "gpt-4")

    # Validate input type
    if not isinstance(input_text, str):
        return jsonify({"error": "Input must be a string"}), 400

    try:
        # Calculate token count using the updated utility function
        token_count = calculate_token_count(input_text, model)
        return jsonify({
            "model": model,
            "tokens": token_count,
            "note": "Token count is estimated for non-OpenAI models" if not model.startswith("gpt-") else "Token count is exact"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to calculate tokens: {str(e)}"}), 500
