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
    
    # 3. Generate the acoustic multi-path environment signature
    print("Synthesizing raw machinery tones, cavitation, and waveguide echoes...")
    received_signal = generate_sonar_signal(t, coords, r_direct, fs=fs)
    
    # 4. Inject low-level background ambient Pink Noise (1/f profile)
    # This acts as our baseline environmental floor before your teammate's stress tests
    print("Injecting ambient background noise floor...")
    white_noise = np.random.normal(0, 1.0, len(t))
    # Fourier transform to scale frequencies by 1/f
    noise_fft = np.fft.rfft(white_noise)
    frequencies = np.fft.rfftfreq(len(t), d=1/fs)
    frequencies[0] = 1.0  # Avoid division by zero at DC component
    pink_filter = 1.0 / np.sqrt(frequencies)
    pink_noise_fft = noise_fft * pink_filter
    pink_noise = np.fft.irfft(pink_noise_fft, len(t))
    
    # Normalize noise energy and mix it into our receiver stream
    pink_noise = (pink_noise / np.std(pink_noise)) * 0.005
    final_sonar_audio = received_signal + pink_noise
    
    # 5. Execute the Short-Time Fourier Transform (STFT)
    print("Computing Short-Time Fourier Transform (STFT)...")
    f_bins, t_bins, Zxx = stft(final_sonar_audio, fs=fs, window='hann', nperseg=256, noverlap=128)
    
    # Convert magnitude to Decibels (dB) for standard sonar visualization
    spectrogram_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
    
    # 6. Plotting the final baseline Spectrogram
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