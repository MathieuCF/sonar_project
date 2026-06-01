import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from data_generation.trajectory import calculate_submarine_trajectory
from data_generation.signal_engine import generate_sonar_signal

def main():
    print("Initializing Arctic Passive Sonar Baseline Simulation...")
    
    # 1. Define global simulation constraints
    fs = 1000.0          # Sampling rate (Hz)
    duration = 60.0      # Total time (seconds)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # 2. Compute the 3D kinematic trajectory path
    print("Calculating target trajectory and 3D coordinate transformations...")
    coords, r_direct = calculate_submarine_trajectory(t)
    
    # 3. Retrieve the singular, composite sea signal from the generation engine
    # (The engine now handles both the target signature and the ambient noise floor)
    print("Receiving environmental acoustic time-series data...")
    final_sonar_audio = generate_sonar_signal(t, coords, r_direct, fs=fs)
    
    # 4. Execute the Short-Time Fourier Transform (STFT) Analysis
    print("Computing Short-Time Fourier Transform (STFT)...")
    f_bins, t_bins, Zxx = stft(final_sonar_audio, fs=fs, window='hann', nperseg=256, noverlap=128)
    
    # Convert magnitude to Decibels (dB) for standard sonar visualization
    spectrogram_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
    
    # 5. Plotting the final baseline Spectrogram
    print("Generating Time-Frequency Spectrogram plot...")
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t_bins, f_bins, spectrogram_db, shading='gouraud', cmap='viridis')
    
    plt.title("Baseline Passive Sonar Spectrogram: Arctic Multiphase Waveguide", fontsize=14, fontweight='bold')
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Frequency (Hz)", fontsize=12)
    plt.ylim(0, 250)  # Zooming into our target region (0 - 250 Hz)
    cbar = plt.colorbar()
    cbar.set_label("Relative Intensity (dB)", fontsize=12)
    
    # Save the output visualization plot
    output_filename = "baseline_sonar_spectrogram.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Simulation complete! Baseline visualization saved as '{output_filename}'")

if __name__ == "__main__":
    main()
