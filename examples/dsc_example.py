import matplotlib.pyplot as plt
from pkynetics.technique_analysis.dsc import DSCExperiment, SpecificHeatCalculator, get_sapphire_cp
from pkynetics.data_import import dsc_importer
import os


# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'pkynetics'))
PKG_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'heat_capacity')


def dsc_analysis_example():
    """Example of DSC specific heat calculation."""

    try:
        print(f"Looking for data in: {PKG_DATA_DIR}")
        dsc_file_sample = os.path.join(PKG_DATA_DIR, 'sample.txt')
        print(f"Looking for the file: {dsc_file_sample}")
        dsc_file_blank = os.path.join(PKG_DATA_DIR, 'zero.txt')
        print(f"Looking for the file: {dsc_file_blank}")
        dsc_file_reference = os.path.join(PKG_DATA_DIR, 'sapphire.txt')
        print(f"Looking for the file: {dsc_file_reference}")

        # Import DSC data
        sample_data = dsc_importer(dsc_file_sample, manufacturer="Setaram")
        blank_data = dsc_importer(dsc_file_blank, manufacturer="Setaram")
        sapphire_data = dsc_importer(dsc_file_reference, manufacturer="Setaram")

        print("DSC data imported successfully.")
        print(sample_data)
        print("Available keys:", sample_data.keys())
        print("Temperature data shape:", sample_data['temperature'].shape)
        print("Time data shape:", sample_data['time'].shape)
        print("Heat flow data shape:", sample_data['heat_flow'].shape)

        # Create experiment objects
        sample_exp = DSCExperiment(
            temperature=sample_data['temperature'],
            heat_flow=sample_data['heat_flow'],
            time=sample_data['time'],
            mass=10.0,  # mg
            name="Sample"
        )

        baseline_exp = DSCExperiment(
            temperature=blank_data['temperature'],
            heat_flow=blank_data['heat_flow'],
            time=blank_data['time'],
            mass=0.0,
            name="Baseline"
        )

        sapphire_exp = DSCExperiment(
            temperature=sapphire_data['temperature'],
            heat_flow=sapphire_data['heat_flow'],
            time=sapphire_data['time'],
            mass=15.0,  # mg
            name="Sapphire"
        )

        # Set temperature range for analysis
        temp_range = (300, 500)  # K
        calculator = SpecificHeatCalculator(temperature_range=temp_range)

        # Calculate Cp using both methods
        cp_two_step, metadata_two = calculator.calculate_two_step(sample_exp, baseline_exp)

        sapphire_cp = get_sapphire_cp(sapphire_exp.temperature)
        cp_three_step, metadata_three = calculator.calculate_three_step(
            sample_exp, sapphire_exp, baseline_exp, sapphire_cp
        )

        # Plot results
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # Plot heat flow signals
        ax1.plot(sample_exp.temperature, sample_exp.heat_flow, label=sample_exp.name)
        ax1.plot(baseline_exp.temperature, baseline_exp.heat_flow, label=baseline_exp.name)
        ax1.plot(sapphire_exp.temperature, sapphire_exp.heat_flow, label=sapphire_exp.name)
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Heat Flow (mW)')
        ax1.set_title('DSC Heat Flow Signals')
        ax1.legend()
        ax1.grid(True)

        # Plot calculated specific heat
        ax2.plot(sample_exp.temperature, cp_two_step, label='Two-step method')
        ax2.plot(sample_exp.temperature, cp_three_step, label='Three-step method')
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Specific Heat (J/g·K)')
        ax2.set_title('Calculated Specific Heat')
        ax2.legend()
        ax2.grid(True)

        # Add text box with results
        textstr = '\n'.join([
            'Two-step method:',
            f'Mean Cp: {metadata_two["mean_cp"]:.3f} J/g·K',
            f'Std Cp: {metadata_two["std_cp"]:.3f} J/g·K',
            '\nThree-step method:',
            f'Mean Cp: {metadata_three["mean_cp"]:.3f} J/g·K',
            f'Std Cp: {metadata_three["std_cp"]:.3f} J/g·K',
        ])
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error in DSC analysis: {str(e)}")
        raise


if __name__ == "__main__":
    dsc_analysis_example()
