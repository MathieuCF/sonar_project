import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import baseline functions from the other files
from data_generation.trajectory import calculate_submarine_trajectory
from data_generation.signal_engine import generate_sonar_signal

# Import animated stress tests and failure animations
from testing_suite.stress_tests import create_noise_animation
from testing_suite.failure_case_sampling import create_sampling_failure_animation


def generate_baseline_spectrogram():
    """Computes and saves the standalone clean baseline spectrogram figure"""
    print("\n[STEP 1/3] Generating Baseline Time-Frequency Spectrogram...")
    
    fs = 1000.0          # Baseline sampling rate (Hz)
    duration = 60.0      # Total tracking duration (seconds)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # 1. Compute trajectory
    coords, r_direct = calculate_submarine_trajectory(t)
    
    # 2. Retrieve the composite sea signal (with target dynamics and base noise floor)
    final_sonar_audio = generate_sonar_signal(t, coords, r_direct, fs=fs)
    
    # 3. Compute short-time fourier transform
    f_bins, t_bins, Zxx = stft(final_sonar_audio, fs=fs, window='hann', nperseg=256, noverlap=128)
    spectrogram_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
    
    # 4. Plot and save
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t_bins, f_bins, spectrogram_db, shading='gouraud', cmap='viridis')
    
    plt.title("Baseline Passive Sonar Spectrogram: Arctic Multiphase Waveguide", fontsize=14, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Frequency (Hz)", fontsize=12)
    plt.ylim(0, 250)
    cbar = plt.colorbar()
    cbar.set_label("Relative Intensity (dB)", fontsize=12)
    
    output_filename = "baseline_sonar_spectrogram.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"-> Success! Baseline Spectrogram visualization saved as '{output_filename}'")


def main():
    print("==================================================================")
    print("   Launching arctic passive sonar...")
    print("==================================================================")
    
    # Phase 1: Clean Baseline Output
    generate_baseline_spectrogram()
    
    # Phase 2: Noise Robustness Animation
    print("\n[STEP 2/3] Executing Noise Robustness Stress Test Loop...")
    create_noise_animation()
    
    # Phase 3: Nyquist Sampling Limitation Animation
    print("\n[STEP 3/3] Executing Downsampling Failure Case Loop...")
    create_sampling_failure_animation()
    
    print("\n==================================================================")
    print("Processing complete.")
    print("The following presentation assets should have been generated in your workspace:")
    print("  - baseline_sonar_spectrogram.png")
    print("  - noise_animation.gif")
    print("  - sampling_failure_animation.gif")
    print("==================================================================")


if __name__ == "__main__":
    main()