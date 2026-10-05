import os
import unittest
import numpy as np
from app import app, evaluate_frame
from database import init_db, log_inference_result, fetch_recent_logs

class TestPoultryStressAdvisor(unittest.TestCase):
    def setUp(self):
        """Set up test environment and initialize database."""
        self.app = app.test_client()
        self.app.testing = True
        init_db()

    def test_homepage_route(self):
        """Verify main web dashboard returns 200 OK."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_missing_video_payload(self):
        """Error Boundary: verify empty payload yields 400 Bad Request."""
        response = self.app.post('/analyze', data={})
        self.assertEqual(response.status_code, 400)

    def test_empty_filename_payload(self):
        """Error Boundary: verify empty filename yields 400 Bad Request."""
        data = {'video': (b'', '')}
        response = self.app.post('/analyze', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)

    def test_frame_stress_evaluation_range(self):
        """Verify image processing pipeline outputs a valid normalized score."""
        dummy_frame = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        score = evaluate_frame(dummy_frame)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 5.0)

    def test_database_logging_and_retrieval(self):
        """Verify database writes and subsequent telemetry queries."""
        log_inference_result(
            frames=4,
            score=3.12,
            severity="MODERATE",
            advisory="Test unit advisory action."
        )
        recent = fetch_recent_logs(limit=1)
        self.assertGreaterEqual(len(recent), 1)
        self.assertEqual(recent[0][3], "MODERATE")

if __name__ == '__main__':
    unittest.main()
