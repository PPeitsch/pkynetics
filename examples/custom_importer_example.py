"""Example usage of CustomImporter class."""

import os
from src.pkynetics.data_import import CustomImporter
import matplotlib.pyplot as plt

# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def custom_import_example():
    """Example of using CustomImporter class."""
    custom_file_path = os.path.join(PROJECT_ROOT, 'data', 'sample_custom_data.csv')

    # Create a sample custom data file
    with open(custom_file_path, 'w') as f:
        f.write("Time(s);Temperature(C);Weight(mg)\n")
        f.write("0;25,0;100,0\n")
        f.write("1;26,5;99,8\n")
        f.write("2;28,0;99,5\n")
        f.write("3;29,5;99,2\n")
        f.write("4;31,0;98,8\n")

    try:
        # Detect delimiter and suggest column names
        delimiter = CustomImporter.detect_delimiter(custom_file_path)
        column_names = CustomImporter.suggest_column_names(custom_file_path, delimiter=delimiter)

        print(f"Detected delimiter: {delimiter}")
        print(f"Suggested column names: {column_names}")

        # Create CustomImporter instance
        importer = CustomImporter(
            custom_file_path,
            column_names,
            separator=delimiter,
            decimal=',',
            skiprows=1  # Skip header row
        )

        # Import data
        data = importer.import_data()

        print("\nImported data:")
        for key, value in data.items():
            print(f"{key}: {value}")

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(data['Time(s)'], data['Temperature(C)'], label='Temperature')
        plt.plot(data['Time(s)'], data['Weight(mg)'], label='Weight')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (°C) / Weight (mg)')
        plt.title('Custom Data Import Example')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Error in custom import example: {str(e)}")

    finally:
        # Remove the sample file
        os.remove(custom_file_path)


if __name__ == "__main__":
    custom_import_example()
