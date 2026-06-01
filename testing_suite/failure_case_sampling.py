import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import imageio.v2 as imageio

# ------------------------------------------------------------
# Allow this script to import files from the main project folder
# ------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from data_generation.trajectory import calculate_submarine_trajectory
from data_generation.signal_engine import generate_sonar_signal


# ------------------------------------------------------------
# Create pink noise
# ------------------------------------------------------------
def create_pink_noise(t, fs, noise_level, seed=42):
    rng = np.random.default_rng(seed)
    white_noise = rng.normal(0, 1.0, len(t))

    noise_fft = np.fft.rfft(white_noise)
    frequencies = np.fft.rfftfreq(len(t), d=1 / fs)
    frequencies[0] = 1.0  # avoid division by zero

    pink_filter = 1.0 / np.sqrt(frequencies)
    pink_noise_fft = noise_fft * pink_filter
    pink_noise = np.fft.irfft(pink_noise_fft, len(t))

    pink_noise = (pink_noise / np.std(pink_noise)) * noise_level
    return pink_noise


# ------------------------------------------------------------
# Calculate alias frequency
# ------------------------------------------------------------
def alias_frequency(true_frequency, fs):
    """
    Calculates where a frequency appears after aliasing.
    Example:
    true frequency = 160 Hz
    fs = 250 Hz
    alias = 90 Hz
    """
    return abs(((true_frequency + fs / 2) % fs) - fs / 2)


# ------------------------------------------------------------
# Compute one spectrogram
# ------------------------------------------------------------
def compute_spectrogram(
    fs=1000.0,
    duration=60.0,
    noise_level=0.005,
    nperseg=256,
    noverlap=128
):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    coords, r_direct = calculate_submarine_trajectory(t)
    received_signal = generate_sonar_signal(t, coords, r_direct, fs=fs)

    pink_noise = create_pink_noise(t, fs, noise_level)
    final_sonar_audio = received_signal + pink_noise

    f_bins, t_bins, Zxx = stft(
        final_sonar_audio,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap
    )

    spectrogram_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
    return t_bins, f_bins, spectrogram_db


# ------------------------------------------------------------
# Save one frame for the sampling failure case
# ------------------------------------------------------------
def save_sampling_failure_frame(
    output_path,
    fs=1000.0,
    duration=60.0,
    noise_level=0.005,
    nperseg=256,
    noverlap=128
):
    t_bins, f_bins, spectrogram_db = compute_spectrogram(
        fs=fs,
        duration=duration,
        noise_level=noise_level,
        nperseg=nperseg,
        noverlap=noverlap
    )

    nyquist = fs / 2
    alias_160 = alias_frequency(160.0, fs)

    plt.figure(figsize=(12, 6))

    plt.pcolormesh(
        t_bins,
        f_bins,
        spectrogram_db,
        shading="gouraud",
        cmap="viridis"
    )

    plt.title(
        f"Failure Case: Sampling Rate Below Nyquist\n"
        f"fs = {fs:.0f} Hz | Nyquist = {nyquist:.0f} Hz | True harmonic = 160 Hz",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Frequency (Hz)", fontsize=12)

    # Only show valid frequency range
    plt.ylim(0, min(250, nyquist))

    cbar = plt.colorbar()
    cbar.set_label("Relative Intensity (dB)", fontsize=12)

    # Mark true 80 Hz tone if it is below Nyquist
    if 80 < nyquist:
        plt.axhline(80, color="white", linestyle="--", linewidth=1.5)
        plt.text(
            1,
            82,
            "True 80 Hz tone",
            fontsize=10,
            color="black",
            backgroundcolor="white"
        )

    # Mark true 160 Hz harmonic if it can be represented
    if 160 < nyquist:
        plt.axhline(160, color="white", linestyle="--", linewidth=1.5)
        plt.text(
            1,
            162,
            "True 160 Hz harmonic",
            fontsize=10,
            color="black",
            backgroundcolor="white"
        )

        status_text = "OK: 160 Hz is below Nyquist and can be represented correctly."

    else:
        # If 160 Hz is above Nyquist, mark where it falsely appears
        plt.axhline(alias_160, color="red", linestyle="--", linewidth=2.0)
        plt.text(
            1,
            alias_160 + 2,
            f"Misleading alias: 160 Hz appears near {alias_160:.1f} Hz",
            fontsize=10,
            color="black",
            backgroundcolor="white"
        )

        status_text = (
            "FAILURE: 160 Hz is above Nyquist.\n"
            "The DFT/STFT can show a false lower frequency."
        )

    # Add explanation box
    plt.text(
        0.98,
        0.95,
        status_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        color="black",
        backgroundcolor="white"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved sampling failure frame: {output_path}")


# ------------------------------------------------------------
# Build GIF from frames
# ------------------------------------------------------------
def build_gif(frame_paths, gif_path, duration_per_frame=0.3):
    print(f"\nBuilding GIF: {gif_path}")
    print(f"Number of frames: {len(frame_paths)}")

    images = []

    for path in frame_paths:
        images.append(imageio.imread(path))

    imageio.mimsave(gif_path, images, duration=duration_per_frame)

    print(f"Saved GIF: {gif_path}")


# ------------------------------------------------------------
# Create sampling failure animation
# ------------------------------------------------------------
def create_sampling_failure_animation():
    print("\nCreating sampling failure / limitation animation...")

    frames_dir = os.path.join(PROJECT_ROOT, "animation_frames", "sampling_failure")
    os.makedirs(frames_dir, exist_ok=True)

    # 50 frames: sampling rate gradually decreases from safe to failing
    fs_values = np.linspace(1000, 200, 50).astype(int)

    frame_paths = []

    for i, fs in enumerate(fs_values):
        fs = float(fs)

        if fs >= 500:
            nperseg = 256
            noverlap = 128
        else:
            nperseg = 128
            noverlap = 64

        frame_path = os.path.join(frames_dir, f"sampling_failure_frame_{i:02d}.png")

        save_sampling_failure_frame(
            output_path=frame_path,
            fs=fs,
            noise_level=0.005,
            nperseg=nperseg,
            noverlap=noverlap
        )

        frame_paths.append(frame_path)

    gif_path = os.path.join(PROJECT_ROOT, "sampling_failure_animation.gif")
    build_gif(frame_paths, gif_path, duration_per_frame=0.3)


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def main():
    print("Starting sampling failure case animation...")

    create_sampling_failure_animation()

    print("\nFailure case animation complete.")
    print("Check the main project folder for:")
    print("- sampling_failure_animation.gif")


if __name__ == "__main__":
    main()