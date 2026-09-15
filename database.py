import sqlite3
from datetime import datetime

DB_NAME = "poultry_analytics.db"

def init_db():
    """Initialize the analytics database schema."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
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

def log_inference_result(frames: int, score: float, severity: str, advisory: str):
    """Persist an inference run to the historical database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO stress_logs (timestamp, processed_frames, average_stress_score, severity_level, advisory)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.utcnow().isoformat(), frames, score, severity, advisory))
    conn.commit()
    conn.close()

def fetch_recent_logs(limit: int = 10):
    """Retrieve the most recent stress assessments."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT timestamp, processed_frames, average_stress_score, severity_level, advisory
        FROM stress_logs
        ORDER BY id DESC
        LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully.")
