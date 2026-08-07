import time
import os
from flask import Blueprint, jsonify, request, current_app
from werkzeug.utils import secure_filename
from backend.config import CLASSES, NUM_CLASSES, UPLOAD_DIR, CONFIDENCE_THRESHOLD
from backend.utils import get_device

main_bp = Blueprint("main", __name__)

@main_bp.route("/", methods=["GET"])
def index():
    return jsonify({
        "project": "LightMamba-ASL: Efficient Video ASL Recognition",
        "status": "online",
        "supported_classes": CLASSES,
        "classes_count": NUM_CLASSES
    })

@main_bp.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": time.time()
    })

@main_bp.route("/api/classes", methods=["GET"])
def get_classes():
    return jsonify({
        "classes": CLASSES
    })

@main_bp.route("/api/model/info", methods=["GET"])
def model_info():
    # We will import prediction service lazily to avoid circular imports during startup
    from backend.services.prediction_service import get_model_details
    details = get_model_details()
    return jsonify(details)

@main_bp.route("/api/predict/video", methods=["POST"])
def predict_video():
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400
        
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty file name"}), 400

    if not file.filename.lower().endswith(".mp4"):
        return jsonify({"success": False, "error": "Only .mp4 format is supported"}), 400

    filename = secure_filename(file.filename)
    filepath = UPLOAD_DIR / f"{int(time.time())}_{filename}"
    file.save(str(filepath))

    try:
        from backend.services.prediction_service import predict_video_file
        start_time = time.time()
        result = predict_video_file(filepath)
        processing_time_ms = (time.time() - start_time) * 1000

        # Clean up temporary file
        if filepath.exists():
            os.remove(filepath)

        return jsonify({
            "success": True,
            "prediction": result["prediction"],
            "confidence": float(result["confidence"]),
            "uncertain": result["uncertain"],
            "top_predictions": result["top_predictions"],
            "processing_time_ms": float(round(processing_time_ms, 2))
        })
    except Exception as e:
        if filepath.exists():
            os.remove(filepath)
        return jsonify({"success": False, "error": f"Inference failed: {str(e)}"}), 500
