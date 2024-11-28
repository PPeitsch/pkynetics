"""Unit tests for CustomImporter class."""

import os
import unittest

import numpy as np
from data_import import CustomImporter

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


class TestCustomImporter(unittest.TestCase):
    def setUp(self):
        self.custom_file_path = os.path.join(PROJECT_ROOT, 'data', 'sample_custom_data.csv')

        # Create a sample custom data file for testing
        with open(self.custom_file_path, 'w') as f:
            f.write("Time(s);Temperature(C);Weight(mg)\n")
            f.write("0;25,0;100,0\n")
            f.write("1;26,5;99,8\n")
            f.write("2;28,0;99,5\n")
            f.write("3;29,5;99,2\n")

    def tearDown(self):
        # Remove the sample file after tests
        os.remove(self.custom_file_path)

    def test_custom_importer(self):
        importer = CustomImporter(
            self.custom_file_path,
            ['Time(s)', 'Temperature(C)', 'Weight(mg)'],
            separator=';',
            decimal=',',
            skiprows=1
        )
        data = importer.import_data()

        self.assertIsInstance(data, dict)
        self.assertIn('Time(s)', data)
        self.assertIn('Temperature(C)', data)
        self.assertIn('Weight(mg)', data)

        self.assertIsInstance(data['Time(s)'], np.ndarray)
        self.assertIsInstance(data['Temperature(C)'], np.ndarray)
        self.assertIsInstance(data['Weight(mg)'], np.ndarray)

        self.assertEqual(len(data['Time(s)']), 4)
        self.assertEqual(len(data['Temperature(C)']), 4)
        self.assertEqual(len(data['Weight(mg)']), 4)

        np.testing.assert_allclose(data['Time(s)'], [0, 1, 2, 3])
        np.testing.assert_allclose(data['Temperature(C)'], [25.0, 26.5, 28.0, 29.5])
        np.testing.assert_allclose(data['Weight(mg)'], [100.0, 99.8, 99.5, 99.2])

    def test_detect_delimiter(self):
        delimiter = CustomImporter.detect_delimiter(self.custom_file_path)
        self.assertEqual(delimiter, ';')

    def test_suggest_column_names(self):
        column_names = CustomImporter.suggest_column_names(self.custom_file_path, delimiter=';')
        self.assertEqual(column_names, ['Time(s)', 'Temperature(C)', 'Weight(mg)'])

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            importer = CustomImporter('non_existent_file.csv', ['time', 'temperature', 'weight'])
            importer.import_data()


if __name__ == '__main__':
    unittest.main()
