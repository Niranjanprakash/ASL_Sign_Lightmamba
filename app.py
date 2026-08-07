from flask import Flask, send_from_directory
from flask_cors import CORS
from backend.routes import main_bp
from backend.utils import verify_paths, get_device
from backend.config import OUTPUT_DIR

def create_app():
    app = Flask(__name__)
    CORS(app)
    
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
