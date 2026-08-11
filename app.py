import os
import urllib.request
from flask import Flask, send_from_directory
from flask_cors import CORS
from backend.routes import main_bp
from backend.utils import verify_paths, get_device
from backend.config import OUTPUT_DIR, CHECKPOINT_DIR

# Download model weights on Render if not present
MODEL_URL = os.environ.get("MODEL_WEIGHTS_URL", "")  # Set this in Render env vars

def download_weights():
    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
    if not checkpoint_path.exists() and MODEL_URL:
        print(f"[STARTUP] Downloading model weights from {MODEL_URL}...")
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, str(checkpoint_path))
        print("[STARTUP] Model weights downloaded.")

def create_app():
    download_weights()  # Download model if running on Render
    app = Flask(__name__)
    CORS(app, origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://niranjanprakash.github.io",  # GitHub Pages frontend
    ])
    
    # Initialize directory paths
    verify_paths()
    
    # Register blueprints
    app.register_blueprint(main_bp)

    # Serve output files (plots, confusion matrix, metrics) for the React frontend
    @app.route('/outputs/<path:filename>')
    def serve_outputs(filename):
        return send_from_directory(str(OUTPUT_DIR), filename)
    
    return app

if __name__ == "__main__":
    # Pre-check device info on startup
    get_device()
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
