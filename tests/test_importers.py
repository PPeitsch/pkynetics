"""Unit tests for the tga, dsc and dilatometry importers."""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import pkynetics
from pkynetics.data_import import dilatometry_importer, dsc_importer, tga_importer
from pkynetics.data_import._manufacturer import detect_manufacturer


class TestImporters(unittest.TestCase):
    def setUp(self):
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()

        # Create sample TGA data
        self.tga_file_path = os.path.join(self.temp_dir, "sample_tga_data.csv")
        tga_data = pd.DataFrame(
            {
                "Time (s)": np.linspace(0, 10, 100),
                "Furnace Temperature (°C)": np.linspace(25, 800, 100),
                "TG (mg)": np.linspace(100, 50, 100),
            }
        )
        tga_data.to_csv(
            self.tga_file_path, sep=";", decimal=",", index=False, encoding="utf-16le"
        )

        # Create sample DSC CSV data
        self.dsc_file_path_csv = os.path.join(self.temp_dir, "sample_dsc_setaram.csv")
        dsc_data = pd.DataFrame(
            {
                "Time (s)": np.linspace(0, 10, 100),
                "Furnace Temperature (°C)": np.linspace(25, 800, 100),
                "Sample Temperature (°C)": np.linspace(25, 790, 100),
                "HeatFlow (mW)": np.random.normal(0, 1, 100),
            }
        )
        dsc_data.to_csv(
            self.dsc_file_path_csv,
            sep=";",
            decimal=",",
            index=False,
            encoding="utf-16le",
        )

        # Create sample DSC TXT data (with header)
        self.dsc_file_path_txt = os.path.join(self.temp_dir, "sample_dsc_setaram.txt")
        with open(self.dsc_file_path_txt, "w", encoding="utf-16le") as f:
            # Write 12 header lines
            for i in range(12):
                f.write(f"Header line {i + 1}\n")
            # Write column names
            f.write(
                "Index;Time (s);Furnace Temperature (°C);Sample Temperature (°C);TG (mg);HeatFlow (mW)\n"
            )
            # Write data
            dsc_data.to_csv(f, sep=";", decimal=",", index=True)

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


# Bundled data from the imported package, so the tests also check that the
# data files ship in the built distribution
PKG_DATA_DIR = os.path.join(os.path.dirname(pkynetics.__file__), "data")
DATA_DIR = os.path.join(PKG_DATA_DIR, "dsc")


class TestEncodingDetection(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.text = "Time (s);Temperature (°C)\n0;25,0\n1;25,5\n" * 20

    def tearDown(self):
        for name in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, name))
        os.rmdir(self.temp_dir)

    def _write(self, data: bytes) -> str:
        path = os.path.join(self.temp_dir, "file.txt")
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_encodings(self):
        from pkynetics.data_import._encoding import detect_encoding

        cases = {
            "utf-16": self.text.encode("utf-16"),  # with BOM
            "utf-16-le": self.text.encode("utf-16-le"),  # without BOM
            "utf-16-be": self.text.encode("utf-16-be"),
            "utf-8-sig": self.text.encode("utf-8-sig"),
            "utf-8": self.text.encode("utf-8"),
            "latin-1": self.text.encode("latin-1"),
        }
        for expected, data in cases.items():
            encoding = detect_encoding(self._write(data))
            self.assertEqual(encoding, expected)
            with open(os.path.join(self.temp_dir, "file.txt"), encoding=encoding) as f:
                self.assertEqual(f.read(), self.text)


class TestTAUniversalAnalysis(unittest.TestCase):
    def test_bundled_eicosane_file(self):
        path = os.path.join(DATA_DIR, "sample_dsc_tainstruments.txt")
        data = dsc_importer(path)  # manufacturer detected from the header

        self.assertEqual(len(data["time"]), 19000)
        # Marker rows with negative time are dropped
        self.assertTrue(np.all(data["time"] >= 0))
        self.assertAlmostEqual(data["time"][0], 187.996)  # min
        self.assertAlmostEqual(data["temperature"][0], -21.18184)  # degC
        self.assertAlmostEqual(data["heat_flow"][0], -10.46185)  # mW
        self.assertIsNone(data["heat_capacity"])


class TestSetaramHeader(unittest.TestCase):
    def test_header_found_by_content(self):
        """Setaram exports: variable number of header lines."""
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, "run.txt")
        header = [
            "Sample - Al 5 steps 58.30mg",
            "Creation Date : 08/01/2025 06:19:50 p.m.",
            "User : admin",
            "",
            "HeatFlow :",
            "  Initial Mass : 58.3 mg",
            "",
        ]
        columns = (
            "Index;Time (s);Furnace Temperature (°C);Sample Temperature (°C);"
            "TG (mg);HeatFlow (mW)"
        )
        rows = [
            f"{i + 1};{i};{98.5 + i};{82.0 + i};240.9;{-17.2 - i}" for i in range(5)
        ]
        with open(path, "w", encoding="utf-16") as f:
            f.write("\n".join(header + [columns] + rows) + "\n")
        try:
            data = dsc_importer(path, manufacturer="Setaram")
        finally:
            os.remove(path)
            os.rmdir(temp_dir)

        np.testing.assert_allclose(data["time"], [0, 1, 2, 3, 4])
        np.testing.assert_allclose(data["sample_temperature"], [82, 83, 84, 85, 86])
        np.testing.assert_allclose(
            data["heat_flow"], [-17.2, -18.2, -19.2, -20.2, -21.2]
        )


class TestDilatometryImporter(unittest.TestCase):
    def test_bundled_files_with_decimal_comma(self):
        """Both bundled .asc exports use a decimal comma (pandas 3: dtype "str")."""
        cases = {
            "sample_dilatometry_data.asc": (73.03217316, 630.20642090),
            "ejemplo_enfriamiento.asc": (2410.00341797, 1049.95727539),
        }
        for name, (time0, temp0) in cases.items():
            with self.subTest(file=name):
                data = dilatometry_importer(os.path.join(PKG_DATA_DIR, name))
                self.assertEqual(
                    set(data),
                    {"time", "temperature", "relative_change", "differential_change"},
                )
                for values in data.values():
                    self.assertEqual(values.dtype, np.float64)
                    self.assertTrue(np.all(np.isfinite(values)))
                self.assertAlmostEqual(data["time"][0], time0)
                self.assertAlmostEqual(data["temperature"][0], temp0)


class TestManufacturerDetection(unittest.TestCase):
    """Every bundled file must import without naming the manufacturer."""

    def test_bundled_dsc_files(self):
        for name in [
            "sample_dsc_setaram.csv",
            "sample_dsc_setaram.txt",
            "sample_dsc_tainstruments.txt",
        ]:
            with self.subTest(file=name):
                data = dsc_importer(os.path.join(DATA_DIR, name))
                self.assertIsNotNone(data["time"])
                self.assertGreater(len(data["time"]), 0)

    def test_bundled_heat_capacity_files(self):
        for name in ["sample.txt", "sapphire.txt", "zero.txt"]:
            with self.subTest(file=name):
                path = os.path.join(PKG_DATA_DIR, "heat_capacity", name)
                data = dsc_importer(path)
                self.assertIsNotNone(data["heat_flow"])

    def test_bundled_tga_file(self):
        data = tga_importer(os.path.join(PKG_DATA_DIR, "sample_tga_data.csv"))
        self.assertIsNotNone(data["weight"])

    def test_setaram_export_without_the_name(self):
        """The whitespace-separated export names no manufacturer, only columns."""
        header = (
            "Duran - MAC250-MS20 34.04mg\n"
            "Creation Date : 26/07/2024 06:03:05 p.m.\n"
            "User : admin\n\n"
            "Index Time       Furnace                  Sample"
            "                  TG         HeatFlow\n"
            "1     0          50.06274                 53.444034"
            "               240.908142 -11.195387\n"
        )
        path = os.path.join(tempfile.mkdtemp(), "run.txt")
        with open(path, "w", encoding="utf-16") as f:
            f.write(header)
        try:
            self.assertEqual(detect_manufacturer(path), "Setaram")
        finally:
            os.remove(path)
            os.rmdir(os.path.dirname(path))

    def test_other_manufacturers_and_unknown(self):
        cases = {
            "TA Instruments Thermal Analysis\nSig1\tTime\nStartOfData\n": "TA",
            "METTLER TOLEDO STARe\n": "Mettler",
            "NETZSCH Proteus\n": "Netzsch",
            "Some other instrument\ncol1,col2\n": None,
        }
        temp_dir = tempfile.mkdtemp()
        for i, (header, expected) in enumerate(cases.items()):
            with self.subTest(expected=expected):
                path = os.path.join(temp_dir, f"run{i}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(header)
                if expected is None:
                    with self.assertRaises(ValueError):
                        detect_manufacturer(path)
                else:
                    self.assertEqual(detect_manufacturer(path), expected)
                os.remove(path)
        os.rmdir(temp_dir)
