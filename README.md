# AI-Driven Poultry Heat Stress Detection & Advisory System

## Architecture Overview
* **Domain Track:** Software / Edge Computer Vision
* **Core Backbone:** OpenCV CLAHE illumination correction + MobileNetV2 feature extraction
* **Telemetry & Storage:** SQLite persistent time-series logging
* **Interface:** Responsive telemetry dashboard with instant local inference

---

## Technical Specifications & Database Schema

### SQLite Database Architecture (`poultry_analytics.db`)

The telemetry engine stores inference logs via the `stress_logs` schema defined in `database.py`:

```sql
CREATE TABLE IF NOT EXISTS stress_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,           -- UTC timestamp in ISO 8601 format
    processed_frames INTEGER NOT NULL, -- Total keyframes extracted and analyzed
    average_stress_score REAL NOT NULL,-- Calculated aggregate score (0.00 - 5.00)
    severity_level TEXT NOT NULL,      -- Classification: NORMAL | MODERATE | CRITICAL
    advisory TEXT NOT NULL             -- Operational advisory text dispatched
);
