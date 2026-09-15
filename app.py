import os
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from database import init_db, log_inference_result, fetch_recent_logs

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize SQLite database on startup
init_db()

# Load MobileNetV2 for posture feature extraction
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def apply_clahe(frame):
    """Enhance low-contrast poultry shed video frames."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def preprocess_frame(frame):
    """Preprocess frame tensor for model evaluation."""
    enhanced = apply_clahe(frame)
    rgb_img = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    return transform(pil_img).unsqueeze(0)

@app.route('/')
def home():
    recent_records = fetch_recent_logs(limit=5)
    return jsonify({
        "status": "online",
        "service": "Poultry Heat Stress Advisory System",
        "version": "2.0-beta",
        "recent_logs": recent_records
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({"error": "No video payload received."}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(file_path)

    cap = cv2.VideoCapture(file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_count = 0
    scores = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Sample 1 frame per second
            if frame_count % fps == 0:
                tensor = preprocess_frame(frame)
                with torch.no_grad():
                    output = model(tensor)
                    score = float(torch.softmax(output, dim=1).max().item() * 5.0)
                    scores.append(score)
            frame_count += 1
    finally:
        cap.release()
        if os.path.exists(file_path):
            os.remove(file_path)

    avg_score = round(sum(scores) / max(len(scores), 1), 2)
    if avg_score < 2.0:
        severity = "NORMAL"
        advisory = "Flock status normal. Routine ventilation active."
    elif avg_score < 3.8:
        severity = "MODERATE"
        advisory = "Moderate stress detected. Trigger circulation fans."
    else:
        severity = "CRITICAL"
        advisory = "High thermal risk! Activate tunnel fans and high-pressure foggers immediately."

    # Persist results in the analytics database
    log_inference_result(len(scores), avg_score, severity, advisory)

    return jsonify({
        "status": "success",
        "processed_frames": len(scores),
        "average_stress_score": avg_score,
        "severity_level": severity,
        "advisory": advisory
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
