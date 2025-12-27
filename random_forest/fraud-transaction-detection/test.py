import unittest

import numpy as np
from sklearn.impute import SimpleImputer


def clean_data(data):
    # Impute missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    data_cleaned = imputer.fit_transform(data)
    return data_cleaned


class TestDataCleaning(unittest.TestCase):
    def test_imputation(self):
        sample_data = np.array([[10], [20], [np.nan]])
        cleaned = clean_data(sample_data)
        # The mean of 10 and 20 is 15. The NaN should become 15.
        self.assertEqual(cleaned[2][0], 15.0)
        self.assertFalse(np.isnan(cleaned).any())


# Run the test
if __name__ == '__main__':
    # Running tests manually for this environment
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataCleaning)
    unittest.TextTestRunner().run(suite)
