"""Unit tests for tga_importer and dsc_importer functions."""

import os
import unittest

import numpy as np
from src.pkynetics.data_import import dsc_importer, tga_importer

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestImporters(unittest.TestCase):
    def setUp(self):
        self.tga_file_path = os.path.join(PROJECT_ROOT, "data", "sample_tga_data.csv")
        self.dsc_file_path = os.path.join(PROJECT_ROOT, "data", "sample_dsc_data.csv")

    def test_tga_importer(self):
        tga_data = tga_importer(self.tga_file_path)

        self.assertIsInstance(tga_data, dict)
        self.assertIn("temperature", tga_data)
        self.assertIn("time", tga_data)
        self.assertIn("weight", tga_data)
        self.assertIn("weight_percent", tga_data)

        self.assertIsInstance(tga_data["temperature"], np.ndarray)
        self.assertIsInstance(tga_data["time"], np.ndarray)
        self.assertIsInstance(tga_data["weight"], np.ndarray)
        self.assertIsInstance(tga_data["weight_percent"], np.ndarray)

        self.assertEqual(len(tga_data["temperature"]), len(tga_data["time"]))
        self.assertEqual(len(tga_data["temperature"]), len(tga_data["weight"]))
        self.assertEqual(len(tga_data["temperature"]), len(tga_data["weight_percent"]))

    def test_dsc_importer(self):
        dsc_data = dsc_importer(self.dsc_file_path)

        self.assertIsInstance(dsc_data, dict)
        self.assertIn("temperature", dsc_data)
        self.assertIn("time", dsc_data)
        self.assertIn("heat_flow", dsc_data)
        self.assertIn("heat_capacity", dsc_data)

        self.assertIsInstance(dsc_data["temperature"], np.ndarray)
        self.assertIsInstance(dsc_data["time"], np.ndarray)
        self.assertIsInstance(dsc_data["heat_flow"], np.ndarray)

        self.assertEqual(len(dsc_data["temperature"]), len(dsc_data["time"]))
        self.assertEqual(len(dsc_data["temperature"]), len(dsc_data["heat_flow"]))

        if dsc_data["heat_capacity"] is not None:
            self.assertIsInstance(dsc_data["heat_capacity"], np.ndarray)
            self.assertEqual(
                len(dsc_data["temperature"]), len(dsc_data["heat_capacity"])
            )

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            tga_importer("non_existent_file.csv")

        with self.assertRaises(FileNotFoundError):
            dsc_importer("non_existent_file.csv")

    def test_invalid_manufacturer(self):
        with self.assertRaises(ValueError):
            tga_importer(self.tga_file_path, manufacturer="InvalidManufacturer")

        with self.assertRaises(ValueError):
            dsc_importer(self.dsc_file_path, manufacturer="InvalidManufacturer")


if __name__ == "__main__":
    unittest.main()
