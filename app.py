from flask import Flask
from routes.embeddings import embeddings_bp
from routes.token_calculation import token_calculation_bp
from qdrant.routes import qdrant_bp
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Register blueprint
app.register_blueprint(embeddings_bp)
app.register_blueprint(token_calculation_bp)
app.register_blueprint(qdrant_bp)


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5005)
