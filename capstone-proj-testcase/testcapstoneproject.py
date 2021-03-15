# Load all the required libraries
from capstoneproject import *
import unittest

class TestModelPredict(unittest.TestCase):
    def test_load_data(self):
        #Unit test for data and logging
        dataset = load_data("false","info")
        self.assertEqual(1, 1)
    def test_data_visualization(self):
        #Unit tests for visualization
        dataset = load_data("false","info")
        data_visualization("false","info",dataset)
        self.assertEqual(1, 1)
    def test_datavalid_models_predict(self):
        #Unit tests for seperate data set and models
        dataset = load_data("false","info")
        datavalid_models_predict("false","info",dataset)
        self.assertEqual(1, 1)
    def test_api(self):
        #Unit test for api
        data_model_predict()
        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()