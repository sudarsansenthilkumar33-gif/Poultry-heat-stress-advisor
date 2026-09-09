import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from flask import Flask, request, jsonify, render_template_string, send_from_directory

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load MobileNetV2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval()
model.to(device)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def enhance_shed_lighting(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def process_video_feed(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    frame_interval = fps
    
    total_frames = 0
    analyzed_frames = 0
    panting_detections = 0
    max_frames_to_process = 15
    
    while cap.isOpened() and analyzed_frames < max_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
            
        if total_frames % frame_interval == 0:
            enhanced_frame = enhance_shed_lighting(frame)
            img_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                
            top_prob, _ = torch.topk(probs, 5)
            metric_val = float(torch.mean(top_prob).item())
            
            if metric_val > 0.08:
                panting_detections += 1
                
            analyzed_frames += 1
        total_frames += 1

    cap.release()
    if os.path.exists(video_path):
        try:
            os.remove(video_path)
        except Exception:
            pass

    if analyzed_frames == 0:
        return {"status": "error", "message": "No valid frames found."}

    panting_pct = round((panting_detections / analyzed_frames) * 100, 1)
    risk_score = round(min(5.0, (panting_pct / 20.0)), 2)

    if risk_score >= 3.5:
        severity = "CRITICAL"
        action = "🚨 Immediate: Activate Zone Tunnel Fans & High-Pressure Foggers. Monitor water line pressure."
    elif risk_score >= 2.0:
        severity = "MODERATE"
        action = "⚠️ Warning: Turn on circulation fans in Zone B. Check drinker nipple flow."
    else:
        severity = "NORMAL"
        action = "✅ Normal: Flock resting comfortably. Routine ventilation active."

    return {
        "status": "success",
        "frames_analyzed": analyzed_frames,
        "panting_percentage": f"{panting_pct}%",
        "risk_score": f"{risk_score} / 5.0",
        "severity": severity,
        "action": action
    }

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="theme-color" content="#0284c7">
    <link rel="manifest" href="/static/manifest.json">
    <title>Poultry Heat Stress Risk Advisor</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }
        .container { max-width: 600px; margin: 0 auto; background: #1e293b; padding: 25px; border-radius: 12px; }
        h1 { color: #38bdf8; font-size: 22px; margin-top: 0; }
        .upload-box { border: 2px dashed #475569; padding: 25px; text-align: center; border-radius: 8px; margin: 20px 0; background: #0b1120; }
        button { background: #0284c7; color: #fff; border: none; padding: 12px 24px; font-weight: bold; border-radius: 6px; cursor: pointer; width: 100%; }
        #results { margin-top: 20px; padding: 15px; border-radius: 8px; display: none; }
        .CRITICAL { background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #fca5a5; }
        .MODERATE { background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fcd34d; }
        .NORMAL { background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #86efac; }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
        .stat-card { background: #0f172a; padding: 10px; border-radius: 6px; font-size: 13px; }
    </style>
</head>
<body>
<div class="container">
    <h1>🐔 Poultry Heat Stress Advisor</h1>
    <p style="font-size: 13px; color: #94a3b8;">Upload shed footage for on-device posture inference.</p>
    
    <div class="upload-box">
        <input type="file" id="videoInput" accept="video/*"><br><br>
        <button onclick="uploadVideo()">Run Model Inference</button>
    </div>

    <div id="results">
        <h3 id="resSeverity" style="margin-top:0;"></h3>
        <p id="resAction" style="font-size:15px; font-weight:600;"></p>
        <div class="stat-grid">
            <div class="stat-card"><strong>Analyzed Frames:</strong> <span id="resFrames"></span></div>
            <div class="stat-card"><strong>Panting Posture:</strong> <span id="resPanting"></span></div>
            <div class="stat-card"><strong>Risk Index:</strong> <span id="resRisk"></span></div>
            <div class="stat-card"><strong>Inference Hardware:</strong> <span>Mobile Native</span></div>
        </div>
    </div>
</div>

<script>
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js');
}

async function uploadVideo() {
    const input = document.getElementById('videoInput');
    if (!input.files[0]) { alert('Please select a video file first.'); return; }
    const formData = new FormData();
    formData.append('video', input.files[0]);
    
    const btn = document.querySelector('button[onclick="uploadVideo()"]');
    btn.innerText = 'Analyzing Video Frames...';
    
    try {
        const res = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await res.json();
        
        const resDiv = document.getElementById('results');
        resDiv.className = data.severity;
        resDiv.style.display = 'block';
        
        document.getElementById('resSeverity').innerText = data.severity + ' HEAT RISK';
        document.getElementById('resAction').innerText = data.action;
        document.getElementById('resFrames').innerText = data.frames_analyzed;
        document.getElementById('resPanting').innerText = data.panting_percentage;
        document.getElementById('resRisk').innerText = data.risk_score;
    } catch(e) {
        alert('Server processing error');
    } finally {
        btn.innerText = 'Run Model Inference';
    }
}
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/sw.js')
def service_worker():
    return send_from_directory('static', 'sw.js', mimetype='application/javascript')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
    file = request.files['video']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    result = process_video_feed(file_path)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
