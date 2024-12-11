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
            'Time (s)': np.linspace(0, 10, 100),
            'Furnace Temperature (°C)': np.linspace(25, 800, 100),
            'TG (mg)': np.linspace(100, 50, 100),
        })
        tga_data.to_csv(self.tga_file_path, sep=';', decimal=',', index=False, encoding='utf-16le')

        # Create sample DSC CSV data
        self.dsc_file_path_csv = os.path.join(self.temp_dir, "sample_dsc_data.csv")
        dsc_data = pd.DataFrame({
            'Time (s)': np.linspace(0, 10, 100),
            'Furnace Temperature (°C)': np.linspace(25, 800, 100),
            'Sample Temperature (°C)': np.linspace(25, 790, 100),
            'HeatFlow (mW)': np.random.normal(0, 1, 100)
        })
        dsc_data.to_csv(self.dsc_file_path_csv, sep=';', decimal=',', index=False, encoding='utf-16le')

        # Create sample DSC TXT data (with header)
        self.dsc_file_path_txt = os.path.join(self.temp_dir, "sample_dsc_data.txt")
        with open(self.dsc_file_path_txt, 'w', encoding='utf-16le') as f:
            # Write 12 header lines
            for i in range(12):
                f.write(f"Header line {i + 1}\n")
            # Write column names
            f.write("Index;Time (s);Furnace Temperature (°C);Sample Temperature (°C);TG (mg);HeatFlow (mW)\n")
            # Write data
            dsc_data.to_csv(f, sep=';', decimal=',', index=True)

    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.tga_file_path):
            os.remove(self.tga_file_path)
        if os.path.exists(self.dsc_file_path_csv):
            os.remove(self.dsc_file_path_csv)
        if os.path.exists(self.dsc_file_path_txt):
            os.remove(self.dsc_file_path_txt)
        os.rmdir(self.temp_dir)

    def test_tga_importer(self):
        tga_data = tga_importer(self.tga_file_path, manufacturer="Setaram")

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

    def test_dsc_importer_csv(self):
        # Test with CSV format
        dsc_data = dsc_importer(self.dsc_file_path_csv, manufacturer="Setaram")

        self.assertIsInstance(dsc_data, dict)
        self.assertIn("temperature", dsc_data)
        self.assertIn("time", dsc_data)
        self.assertIn("heat_flow", dsc_data)
        self.assertIn("sample_temperature", dsc_data)

        self.assertIsInstance(dsc_data["temperature"], np.ndarray)
        self.assertIsInstance(dsc_data["time"], np.ndarray)
        self.assertIsInstance(dsc_data["heat_flow"], np.ndarray)
        self.assertIsInstance(dsc_data["sample_temperature"], np.ndarray)

        self.assertEqual(len(dsc_data["temperature"]), len(dsc_data["time"]))
        self.assertEqual(len(dsc_data["temperature"]), len(dsc_data["heat_flow"]))

    def test_dsc_importer_txt(self):
        # Test with TXT format
        dsc_data = dsc_importer(self.dsc_file_path_txt, manufacturer="Setaram")

        self.assertIsInstance(dsc_data, dict)
        self.assertIn("temperature", dsc_data)
        self.assertIn("time", dsc_data)
        self.assertIn("heat_flow", dsc_data)
        self.assertIn("sample_temperature", dsc_data)

        self.assertIsInstance(dsc_data["temperature"], np.ndarray)
        self.assertIsInstance(dsc_data["time"], np.ndarray)
        self.assertIsInstance(dsc_data["heat_flow"], np.ndarray)
        self.assertIsInstance(dsc_data["sample_temperature"], np.ndarray)

        self.assertEqual(len(dsc_data["temperature"]), len(dsc_data["time"]))
        self.assertEqual(len(dsc_data["temperature"]), len(dsc_data["heat_flow"]))

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            tga_importer("non_existent_file.csv")

        with self.assertRaises(FileNotFoundError):
            dsc_importer("non_existent_file.csv")

    def test_invalid_manufacturer(self):
        with self.assertRaises(ValueError):
            tga_importer(self.tga_file_path, manufacturer="InvalidManufacturer")

        with self.assertRaises(ValueError):
            dsc_importer(self.dsc_file_path_csv, manufacturer="InvalidManufacturer")


if __name__ == "__main__":
    unittest.main()
