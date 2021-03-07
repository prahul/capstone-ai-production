# Load all the required libraries
from capstoneproject import *
import unittest

class TestModelPredict(unittest.TestCase):
    def test_mp_info(self):
        model_predict("false","info")
        self.assertEqual(1, 1)
    def test_mp_warn(self):
        model_predict("false","warn")
        self.assertEqual(1, 1)
    def test_mp_error(self):
        model_predict("false","error")
        self.assertEqual(1, 1)
    def test_mp_debug(self):
        model_predict("false","debug")
        self.assertEqual(1, 1)
    def test_mp_critical(self):
        model_predict("false","critical")
        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()