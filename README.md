# AI-Driven Poultry Heat Stress Detection & Advisory System

## Project Review-1 Milestone (35% Scope)
**Track:** Software / Edge AI  
**Deployment:** Flask Progressive Web App (PWA) with MobileNetV2 Posture Inference  

---

### 1. What Has Been Completed So Far (35% Target)
* **Neural Inference Engine:** Integrated a lightweight PyTorch **MobileNetV2** backbone configured for on-device poultry shed video inference.
* **Computer Vision Preprocessing:** Implemented OpenCV-based **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to normalize variable illumination inside poultry sheds.
* **Backend Architecture:** Built a **Flask REST API** to handle asynchronous video upload, temporal frame subsampling (1 fps extraction), and posture anomaly scoring.
* **Progressive Web App (PWA):** Developed an offline-capable, responsive dashboard with `manifest.json` and `sw.js` allowing direct mobile installation.

---

### 2. Key Modules & Features Completed
* **Frame Subsampling Engine:** Temporal interval processing to ensure sub-5 second response latency on local hardware.
* **Posture Heat Risk Index:** Automated mapping of flock panting percentage to a 5-point heat stress risk index (0.0 to 5.0).
* **Tiered Action Recommender:** Automated operational alerts:
  * **NORMAL:** Routine shed ventilation.
  * **MODERATE:** Zone B circulation fan trigger.
  * **CRITICAL:** High-pressure foggers and zone tunnel fan activation.
* **Network Bridging:** Multi-device access enabled via local network broadcast (`0.0.0.0:5000`).

---

### 3. What Is Currently Working
* Full pipeline flow: **Video Upload -> CLAHE Normalization -> MobileNetV2 Inference -> Posture Analysis -> Action Advisory UI**.
* Standalone PWA installation workflow across desktop and Android mobile devices.

---

### 4. Pending Work & Roadmap (Remaining 65% - Software Track)
* **Model Optimization (Review-2):** Quantization and conversion of PyTorch weights to ONNX Runtime / WebAssembly for fast client/edge execution.
* **Batch Video Streaming:** Asynchronous queue processing for handling concurrent multi-camera shed feeds.
* **Historical Analytics Database:** SQLite/PostgreSQL schema integration to track flock heat index trends over time.
* **Automated Notification Dispatcher:** Webhook/Email alert system for immediate notification during high-risk thresholds.

---

## Technical Architecture & API Documentation

### REST API Endpoints

#### 1. Dashboard Shell & Service Worker
* **Endpoint:** `GET /`
* **Description:** Delivers the responsive Progressive Web App dashboard shell and client service worker.
* **Response:** `200 OK` (HTML)

#### 2. Video Anomaly Processing
* **Endpoint:** `POST /analyze`
* **Payload:** `multipart/form-data` containing `video` file (.mp4, .avi).
* **Response Format (`application/json`):**
```json
{
  "status": "success",
  "processed_frames": 12,
  "average_stress_score": 3.45,
  "severity_level": "MODERATE",
  "advisory": "Trigger circulation fans and verify shed water distribution line pressure."
}
