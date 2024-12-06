"""Unit tests for tga_importer and dsc_importer functions."""

import os
import unittest
import tempfile
import numpy as np
import pandas as pd

from src.pkynetics.data_import import dsc_importer, tga_importer


class TestImporters(unittest.TestCase):
    def setUp(self):
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample TGA data
        self.tga_file_path = os.path.join(self.temp_dir, "sample_tga_data.csv")
        tga_data = pd.DataFrame({
            'Temperature (°C)': np.linspace(25, 800, 100),
            'Time (min)': np.linspace(0, 100, 100),
            'Weight (mg)': np.linspace(100, 50, 100),
            'Weight (%)': np.linspace(100, 50, 100)
        })
        tga_data.to_csv(self.tga_file_path, index=False)
        
        # Create sample DSC data
        self.dsc_file_path = os.path.join(self.temp_dir, "sample_dsc_data.csv")
        dsc_data = pd.DataFrame({
            'Temperature (°C)': np.linspace(25, 800, 100),
            'Time (min)': np.linspace(0, 100, 100),
            'Heat Flow (mW)': np.random.normal(0, 1, 100),
            'Heat Capacity (J/(g·°C))': np.random.normal(1, 0.1, 100)
        })
        # Add TA Instruments header
        with open(self.dsc_file_path, 'w') as f:
            f.write("TA Instruments Thermal Analysis\n")
        dsc_data.to_csv(self.dsc_file_path, mode='a', index=False)

    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.tga_file_path):
            os.remove(self.tga_file_path)
        if os.path.exists(self.dsc_file_path):
            os.remove(self.dsc_file_path)
        os.rmdir(self.temp_dir)

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
            self.assertEqual(len(dsc_data["temperature"]), len(dsc_data["heat_capacity"]))

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
