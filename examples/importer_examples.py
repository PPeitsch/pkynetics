"""Examples for using tga_importer, dsc_importer, and dilatometry_importer functions."""

import os
from data_import import tga_importer, dsc_importer, dilatometry_importer
import time

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def tga_import_example():
    """Example of using tga_importer function."""
    tga_file_path = os.path.join(PROJECT_ROOT, 'data', 'sample_tga_data.csv')
    try:
        tga_data = tga_importer(file_path=tga_file_path, manufacturer="Setaram")
        print("TGA data imported successfully.")
        print("Available keys:", tga_data.keys())
        print("Temperature data shape:", tga_data['temperature'].shape)
        print("Time data shape:", tga_data['time'].shape)
        print("Weight data shape:", tga_data['weight'].shape)
        print("Weight percent data shape:", tga_data['weight_percent'].shape)
    except Exception as e:
        print(f"Error importing TGA data: {str(e)}")


def dsc_import_example():
    """Example of using dsc_importer function."""
    dsc_file_path = os.path.join(PROJECT_ROOT, 'data', 'sample_dsc_data.txt')
    try:
        dsc_data = dsc_importer(file_path=dsc_file_path, manufacturer="Setaram")
        print("DSC data imported successfully.")
        print("Available keys:", dsc_data.keys())
        print("Temperature data shape:", dsc_data['temperature'].shape)
        print("Time data shape:", dsc_data['time'].shape)
        print("Heat flow data shape:", dsc_data['heat_flow'].shape)
        if dsc_data['heat_capacity'] is not None:
            print("Heat capacity data shape:", dsc_data['heat_capacity'].shape)
        else:
            print("Heat capacity data not available.")
    except Exception as e:
        print(f"Error importing DSC data: {str(e)}")


def dilatometry_import_example():
    """Example of using dilatometry_importer function."""
    dilatometry_file_path = os.path.join(PROJECT_ROOT, 'data', 'sample_dilatometry_data.asc')
    try:
        dilatometry_data = dilatometry_importer(dilatometry_file_path)
        print("Dilatometry data imported successfully.")
        print("Available keys:", dilatometry_data.keys())
        print("Time data shape:", dilatometry_data['time'].shape)
        print("Temperature data shape:", dilatometry_data['temperature'].shape)
        print("Relative change data shape:", dilatometry_data['relative_change'].shape)
        print("Differential change data shape:", dilatometry_data['differential_change'].shape)

        # Print first few rows of each data type
        print("\nFirst 5 rows of time data:", dilatometry_data['time'][:5])
        print("First 5 rows of temperature data:", dilatometry_data['temperature'][:5])
        print("First 5 rows of relative change data:", dilatometry_data['relative_change'][:5])
        print("First 5 rows of differential change data:", dilatometry_data['differential_change'][:5])
    except Exception as e:
        print(f"Error importing dilatometry data: {str(e)}")


if __name__ == "__main__":
    print("Running TGA import example:")
    tga_import_example()
    time.sleep(1)
    print("\nRunning DSC import example:")
    dsc_import_example()
    time.sleep(1)
    print("\nRunning Dilatometry import example:")
    dilatometry_import_example()