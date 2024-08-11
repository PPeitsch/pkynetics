"""Examples for using tga_importer and dsc_importer functions."""

import os
from data_import import tga_importer, dsc_importer

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def tga_import_example():
    """Example of using tga_importer function."""
    tga_file_path = os.path.join(PROJECT_ROOT, 'data', 'sample_tga_data.csv')
    try:
        tga_data = tga_importer(tga_file_path)
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
    dsc_file_path = os.path.join(PROJECT_ROOT, 'data', 'sample_dsc_data.csv')
    try:
        dsc_data = dsc_importer(dsc_file_path)
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


if __name__ == "__main__":
    print("Running TGA import example:")
    tga_import_example()
    print("\nRunning DSC import example:")
    dsc_import_example()
