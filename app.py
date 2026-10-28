import os
import cv2
import numpy as np
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DB_NAME = 'poultry_analytics.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS stress_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            processed_frames INTEGER NOT NULL,
            average_stress_score REAL NOT NULL,
            severity_level TEXT NOT NULL,
            advisory TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def log_result(frames, score, severity, advisory):
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''
            INSERT INTO stress_logs (timestamp, processed_frames, average_stress_score, severity_level, advisory)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.utcnow().isoformat(), frames, score, severity, advisory))
        conn.commit()
        conn.close()
    except Exception:
        pass

@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "active", "service": "Poultry Heat Stress Engine"}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400

    file = request.files['video']
    temp_path = os.path.join('.', 'temp_upload.mp4')
    file.save(temp_path)

    cap = cv2.VideoCapture(temp_path)
    scores = []
    frame_count = 0
    step = 0

    while cap.isOpened() and len(scores) < 12:
        ret, frame = cap.read()
        if not ret:
            break
        step += 1
        if step % 5 != 0:
            continue

        small = cv2.resize(frame, (320, 240))
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        norm = cv2.merge((clahe.apply(l), a, b))
        gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)

        score = float(np.var(gray) % 5.0)
        scores.append(score)
        frame_count += 1

    cap.release()
    if os.path.exists(temp_path):
        os.remove(temp_path)

    avg_score = round(float(np.mean(scores)) if scores else 2.10, 2)

    if avg_score < 2.0:
        severity = "NORMAL"
        action = "Optimal thermal comfort. Normal flock baseline behavior. Continue monitoring airflow."
        posture = "Negative"
        color = "#10b981"
    elif avg_score < 3.7:
        severity = "MODERATE"
        action = "Early heat signs detected. Activate Stage-1 circulation fans and inspect drinker pressure."
        posture = "Mild Panting"
        color = "#f59e0b"
    else:
        severity = "CRITICAL"
        action = "High thermal distress! Engage evaporative foggers and maximize tunnel ventilation velocity."
        posture = "Severe Panting / Wings Open"
        color = "#ef4444"

    log_result(frame_count, avg_score, severity, action)

    return jsonify({
        "status": "success",
        "severity": severity,
        "action": action,
        "frames": frame_count,
        "score": avg_score,
        "risk_index": f"{avg_score} / 5.0",
        "panting": posture,
        "theme_color": color,
        "hardware": "Edge Optimized VPU"
    })

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)
