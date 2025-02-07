"""Example of DSC analysis with segment selection and improved error handling."""

import os
import logging
from typing import Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from pkynetics.data_import import dsc_importer
from pkynetics.technique_analysis.dsc.types import DSCExperiment
from pkynetics.technique_analysis.dsc.baseline import BaselineCorrector
from pkynetics.technique_analysis.dsc.peak_analysis import PeakAnalyzer
from pkynetics.technique_analysis.dsc.thermal_events import ThermalEventDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PKG_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "pkynetics", "data", "dsc")


class DSCSegmentAnalyzer:
    def __init__(self, experiment: DSCExperiment, manual_range: Optional[Tuple[float, float]] = None):
        self.experiment = experiment
        self.segments = []
        if manual_range is not None:
            self._create_manual_segment(manual_range)
        else:
            self._detect_segments()

    def _create_manual_segment(self, temp_range: Tuple[float, float]) -> None:
        """Create segment from manual temperature range."""
        start_temp, end_temp = temp_range
        start_idx = np.abs(self.experiment.temperature - start_temp).argmin()
        end_idx = np.abs(self.experiment.temperature - end_temp).argmin()

        if end_idx - start_idx > 100:
            self.segments = [(start_idx, end_idx)]
        else:
            raise ValueError("Selected range too small (minimum 100 points required)")

    def _detect_segments(self) -> None:
        """Detect different segments in the DSC curve."""
        # Calculate derivative to find rate changes
        dT = np.gradient(self.experiment.temperature)
        rate = np.gradient(self.experiment.temperature, self.experiment.time)

        # Find significant rate changes
        rate_changes = find_peaks(np.abs(np.gradient(rate)))[0]

        if len(rate_changes) == 0:
            # If no clear segments, use the whole curve
            self.segments = [(0, len(self.experiment.temperature) - 1)]
            return

        # Create segments
        start_idx = 0
        for end_idx in rate_changes:
            if end_idx - start_idx > 100:  # Minimum segment size
                self.segments.append((start_idx, end_idx))
            start_idx = end_idx

        # Add final segment
        if len(self.experiment.temperature) - start_idx > 100:
            self.segments.append((start_idx, len(self.experiment.temperature) - 1))

    def plot_segments(self) -> None:
        """Plot identified segments for user selection."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.experiment.temperature, self.experiment.heat_flow, 'b-', alpha=0.5)

        for i, (start, end) in enumerate(self.segments):
            plt.plot(
                self.experiment.temperature[start:end],
                self.experiment.heat_flow[start:end],
                label=f'Segment {i + 1}'
            )

        plt.xlabel('Temperature (K)')
        plt.ylabel('Heat Flow (mW)')
        plt.title('DSC Curve Segments')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_segment(self, segment_idx: int) -> DSCExperiment:
        """Get a specific segment as a new DSCExperiment object."""
        if not 0 <= segment_idx < len(self.segments):
            raise ValueError(f"Invalid segment index. Available segments: 0-{len(self.segments) - 1}")

        start, end = self.segments[segment_idx]

        return DSCExperiment(
            temperature=self.experiment.temperature[start:end],
            heat_flow=self.experiment.heat_flow[start:end],
            time=self.experiment.time[start:end],
            mass=self.experiment.mass,
            heating_rate=self.experiment.heating_rate,
            sample_name=f"{self.experiment.sample_name}_segment_{segment_idx}"
        )


def analyze_segment(segment: DSCExperiment) -> Dict:
    """Analyze a single DSC segment."""
    results = {}

    try:
        # Validate data
        if len(segment.temperature) < 100:
            raise ValueError("Segment too short for analysis")

        if not np.all(np.diff(segment.temperature) > 0):
            logger.warning("Temperature not strictly increasing, sorting data...")
            sort_idx = np.argsort(segment.temperature)
            segment.temperature = segment.temperature[sort_idx]
            segment.heat_flow = segment.heat_flow[sort_idx]
            segment.time = segment.time[sort_idx]

        # Baseline correction
        baseline_corrector = BaselineCorrector(smoothing_window=min(21, len(segment.temperature) // 5))
        baseline_result = baseline_corrector.correct(
            segment.temperature,
            segment.heat_flow,
            method='linear'
        )
        results['baseline'] = baseline_result

        # Peak analysis
        peak_analyzer = PeakAnalyzer()
        peaks = peak_analyzer.find_peaks(
            segment.temperature,
            baseline_result.corrected_data,
            baseline_result.baseline
        )
        results['peaks'] = peaks

        # Event detection
        if len(peaks) > 0:
            event_detector = ThermalEventDetector()
            events = event_detector.detect_events(
                segment.temperature,
                baseline_result.corrected_data,
                peaks,
                baseline_result.baseline
            )
            results['events'] = events

    except Exception as e:
        logger.error(f"Error analyzing segment: {str(e)}")
        results['error'] = str(e)

    return results


def plot_segment_analysis(segment: DSCExperiment, results: Dict) -> None:
    """Plot analysis results for a segment."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Original data and baseline
    ax1.plot(segment.temperature, segment.heat_flow, 'b-', label='Original')
    if 'baseline' in results:
        ax1.plot(segment.temperature, results['baseline'].baseline, 'r--', label='Baseline')
        ax1.plot(segment.temperature, results['baseline'].corrected_data, 'g-', label='Corrected')
    ax1.set_xlabel('Temperature (K)')
    ax1.set_ylabel('Heat Flow (mW)')
    ax1.legend()
    ax1.grid(True)

    # Peaks and events
    if 'peaks' in results:
        for peak in results['peaks']:
            ax2.axvline(peak.peak_temperature, color='r', linestyle='--', alpha=0.5)
            ax2.axvline(peak.onset_temperature, color='g', linestyle=':', alpha=0.5)
            ax2.axvline(peak.endset_temperature, color='g', linestyle=':', alpha=0.5)
    ax2.plot(segment.temperature, segment.heat_flow, 'b-')
    ax2.set_xlabel('Temperature (K)')
    ax2.set_ylabel('Heat Flow (mW)')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    """Main execution function."""
    # File path
    file_path = os.path.join(PKG_DATA_DIR, "sample_dsc_setaram.txt")

    try:
        # Load data
        data = dsc_importer(file_path=file_path, manufacturer="Setaram")
        experiment = DSCExperiment(
            temperature=data['temperature'],
            heat_flow=data['heat_flow'],
            time=data['time'],
            mass=10.0,  # Example mass in mg
            heating_rate=10.0  # Example heating rate in K/min
        )

        # Initialize segment analyzer
        use_manual = input("Use manual temperature range? (y/n): ").lower() == 'y'

        if use_manual:
            start_temp = float(input("Enter start temperature (K): "))
            end_temp = float(input("Enter end temperature (K): "))
            segment_analyzer = DSCSegmentAnalyzer(experiment, manual_range=(start_temp, end_temp))
        else:
            segment_analyzer = DSCSegmentAnalyzer(experiment)

        # Show available segments
        print(f"\nFound {len(segment_analyzer.segments)} segments")
        segment_analyzer.plot_segments()

        # Analyze each segment
        for i in range(len(segment_analyzer.segments)):
            print(f"\nAnalyzing segment {i + 1}...")
            segment = segment_analyzer.get_segment(i)
            results = analyze_segment(segment)

            if 'error' not in results:
                plot_segment_analysis(segment, results)

                if 'peaks' in results:
                    print(f"\nFound {len(results['peaks'])} peaks in segment {i + 1}:")
                    for j, peak in enumerate(results['peaks']):
                        print(f"\nPeak {j + 1}:")
                        print(f"Onset Temperature: {peak.onset_temperature:.2f} K")
                        print(f"Peak Temperature: {peak.peak_temperature:.2f} K")
                        print(f"Endset Temperature: {peak.endset_temperature:.2f} K")
                        print(f"Enthalpy: {peak.enthalpy:.2f} J/g")
            else:
                print(f"Error analyzing segment {i + 1}: {results['error']}")

    except Exception as e:
        logger.error(f"Error in DSC analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
