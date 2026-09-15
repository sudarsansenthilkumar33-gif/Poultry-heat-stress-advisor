import unittest
import numpy as np
from app import app, preprocess_frame

class TestPoultryAdvisor(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_homepage_route(self):
        """Test if the main dashboard loads successfully."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_invalid_file_upload(self):
        """Test error boundary for missing or invalid video payload."""
        response = self.app.post('/analyze', data={})
        self.assertIn(response.status_code, [400, 422, 500])

    def test_clahe_preprocessing_shape(self):
        """Test CLAHE image pipeline output dimensions."""
        dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        processed_tensor = preprocess_frame(dummy_frame)
        self.assertEqual(processed_tensor.shape, (1, 3, 224, 224))

if __name__ == '__main__':
    unittest.main()
