"""Reproducibility checks for the public-data model workflow."""

import unittest

import numpy as np

from src.data import load_heart_data
from src.modeling import explain_case, train_experiment


class ProjectPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_heart_data()

    def test_data_shape_and_target_mapping(self):
        self.assertEqual(self.X.shape, (303, 13))
        self.assertEqual(set(self.y.unique()), {0, 1})
        self.assertTrue(self.X.isna().any().any())

    def test_training_and_prediction_pipeline(self):
        result = train_experiment(self.X, self.y)
        self.assertEqual(len(result["y_test"]), 61)
        self.assertEqual(result["confusion_matrix"].shape, (2, 2))
        self.assertIn(result["best_name"], result["models"])
        probabilities = result["best_model"].predict_proba(result["X_test"])
        self.assertEqual(probabilities.shape, (61, 2))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_local_shap_output_preserves_original_features(self):
        result = train_experiment(self.X, self.y)
        row = result["X_test"].iloc[[0]]
        explanation = explain_case(result, row, max_evals=50)
        self.assertEqual(set(explanation["Feature"]), set(self.X.columns))
        self.assertTrue(np.isfinite(explanation["SHAP value"]).all())


if __name__ == "__main__":
    unittest.main()
