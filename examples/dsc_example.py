import matplotlib.pyplot as plt
from technique_analysis.dsc import DSCExperiment, SpecificHeatCalculator, get_sapphire_cp
from data_import import dsc_importer
import os


def dsc_analysis_example():
    """Example of DSC specific heat calculation."""
    # Get project root directory
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'heat_capacity')

    try:
        # Import DSC data
        sample_data = dsc_importer(
            os.path.join(DATA_DIR, 'sample.txt'),
            manufacturer="Setaram")
        baseline_data = dsc_importer(
            os.path.join(DATA_DIR, 'zero.txt'),
            manufacturer="Setaram")
        sapphire_data = dsc_importer(
            os.path.join(DATA_DIR, 'sapphire.txt'),
            manufacturer="Setaram")

        # Create experiment objects
        sample_exp = DSCExperiment(
            temperature=sample_data['temperature'],
            heat_flow=sample_data['heat_capacity'],
            time=sample_data['time'],
            mass=10.0,  # mg
            name="Sample"
        )

        baseline_exp = DSCExperiment(
            temperature=baseline_data['temperature'],
            heat_flow=baseline_data['heat_capacity'],
            time=baseline_data['time'],
            mass=0.0,
            name="Baseline"
        )

        sapphire_exp = DSCExperiment(
            temperature=sapphire_data['temperature'],
            heat_flow=sapphire_data['heat_capacity'],
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
