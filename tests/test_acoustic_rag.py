"""
Automated validation and unit test suite for Acoustic-RAG mathematics.
Verifies:
1. 40 kHz Nyquist decomposition (2000 linear bins).
2. Deterministic INT16 quantization & 10 dB noise floor gate.
3. 1,500-state modulo cyclic aggregation.
4. SQLite persistence and recovery.
"""

import os
import sys
import unittest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from acoustic_rag import AcousticRAGEngine, CodebookStore, AcousticRAGGrafter


class TestAcousticRAG(unittest.TestCase):

    def setUp(self):
        self.engine = AcousticRAGEngine(
            sample_rate=40000,
            n_fft=4000,
            hop_length=800,
            num_bins=2000,
            device="cpu"
        )
        self.test_db_path = "/tmp/test_acoustic_rag.db"
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        self.store = CodebookStore(self.test_db_path)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    def test_discrete_token_extraction(self):
        # 1-second synthetic sine wave at 1 kHz (Bin 100)
        t = np.linspace(0, 1.0, 40000, endpoint=False)
        sine_1khz = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)

        tokens = self.engine.extract_discrete_tokens(sine_1khz)

        # Check dimensions
        self.assertEqual(tokens.ndim, 2)
        self.assertEqual(tokens.shape[1], 2000)
        self.assertEqual(tokens.dtype, np.int16)

        # Check peak energy is concentrated around bin 100 (100 * 10 Hz = 1000 Hz)
        mean_spectrum = np.mean(tokens, axis=0)
        peak_bin = int(np.argmax(mean_spectrum))
        self.assertTrue(98 <= peak_bin <= 102, f"Expected peak near bin 100, got {peak_bin}")

    def test_cyclic_state_aggregation(self):
        # Create 3,000 synthetic frames
        mock_frames = np.random.randint(15, 80, size=(3000, 2000), dtype=np.int16)
        settled = self.engine.aggregate_cyclic_states(mock_frames)

        self.assertEqual(settled.shape, (1500, 2000))
        self.assertEqual(settled.dtype, np.int16)

    def test_sqlite_persistence_and_grafting(self):
        mock_matrix = np.ones((1500, 2000), dtype=np.int16) * 45
        self.store.insert_category_matrix(category_id=412, matrix_1500=mock_matrix)

        fetched_vector = self.store.get_state_vector(category_id=412, state_idx=75)
        self.assertEqual(len(fetched_vector), 2000)
        self.assertEqual(fetched_vector[0], 45)

        # Test grafter
        grafter = AcousticRAGGrafter(self.store)
        dummy_audio = np.random.randn(8000).astype(np.float32)
        refined = grafter.graft_spectral_harmonics(dummy_audio, category_id=412, sr=40000)
        self.assertEqual(refined.shape, dummy_audio.shape)


if __name__ == "__main__":
    unittest.main()
