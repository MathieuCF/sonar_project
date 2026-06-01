import numpy as np
from scipy.signal import lfilter

def generate_sonar_signal(t, coords, r_direct, fs=1000.0):
    """
    Computes and synthesizes a realistic dynamic sound signal combining background pink noise and 
    pulse modulated propeller cavitation sound.
    
    Parameters:

    t : ndarray
        1D NumPy array representing the time steps of the simulation (seconds).
    coords : dict
        Dictionary containing arrays for 'x', 'y', 'z' positions of the sub over time.
    r_direct : ndarray
        1D array containing the true direct line-of-sight distance (meters).
    fs : float
        Sampling rate of the simulation system (Hz).
        
    Returns:

    composite_sea_signal : ndarray
        1D array representing the singular composite voltage/pressure acoustic stream 
        received at the stationary hydrophone sensor.
    """
    N = len(t)
    c = 1440.0  # Speed of sound in cold Arctic water (m/s)
    
    # -------------------------------------------------------------------------
    # PART 1: SYNTHESIZE RAW SUBMARINE SIGNATURE (Machinery + Cavitation)
    # -------------------------------------------------------------------------
    # Machinery noise (80 Hz + 160 Hz harmonic)
    machinery_tone = np.sin(2 * np.pi * 80.0 * t) + 0.4 * np.sin(2 * np.pi * 160.0 * t)
    
    # Broadband propeller cavitation (White noise shaped into broadband hiss)
    raw_hiss = np.random.normal(0, 1, N)
    # Simple low-pass filter to shape the broadband cavitation curve
    b, a = [0.1], [1, -0.9] 
    shaped_hiss = lfilter(b, a, raw_hiss)
    
    # modulating a imagined 7 blade propeller (sine wave modulation)
    blade_rate = 1.5  # Hz
    modulation = 1.0 + 0.6 * np.sin(2 * np.pi * blade_rate * t)
    cavitation_signature = shaped_hiss * modulation
    
    # Combine into a single core structural signature emitted by the boat
    raw_sub_signal = 1.5 * machinery_tone + 1.0 * cavitation_signature
    

    # Hydrophone depth is fixed at 200m in a 400m water column
    z_hydro = 200.0
    H = 400.0
    
    # Initialize our intermediate hydrophone receiver array for target acoustic energy
    received_signal = np.zeros(N)
    
    # Vectorized loop through time to apply dynamic propagation delays
    for i in range(N):
        # Current 3D position of the sub
        x, y, z = coords['x'][i], coords['y'][i], coords['z'][i]
        
        # 1. Direct Path
        r_d = r_direct[i]
        t_delay_direct = r_d / c
        idx_d = i - int(t_delay_direct * fs)
        
        # 2. Ice Ceiling Reflection Path (Bounces off z=0)
        r_ice = np.sqrt(x**2 + y**2 + (z + z_hydro)**2)
        t_delay_ice = r_ice / c
        idx_ice = i - int(t_delay_ice * fs)
        
        # 3. Seafloor Reflection Path (Bounces off z=H)
        r_floor = np.sqrt(x**2 + y**2 + ((H - z) + (H - z_hydro))**2)
        t_delay_floor = r_floor / c
        idx_floor = i - int(t_delay_floor * fs)
        
        # Constructive/destructive accumulation with environmental damping (attenuation / r)
        if idx_d >= 0:
            received_signal[i] += raw_sub_signal[idx_d] / r_d
        if idx_ice >= 0:
            received_signal[i] += (-0.4 * raw_sub_signal[idx_ice]) / r_ice
        if idx_floor >= 0:
            received_signal[i] += (0.6 * raw_sub_signal[idx_floor]) / r_floor
            
    # -------------------------------------------------------------------------
    # PART 2: AMBIENT BACKGROUND NOISE GENERATION (1/f Pink Noise Floor)
    # -------------------------------------------------------------------------
    white_noise = np.random.normal(0, 1.0, N)
    
    noise_fft = np.fft.rfft(white_noise)
    frequencies = np.fft.rfftfreq(N, d=1/fs)
    frequencies[0] = 1.0  # Safeguard against division by zero at DC
    
    pink_filter = 1.0 / np.sqrt(frequencies)
    pink_noise_fft = noise_fft * pink_filter
    pink_noise = np.fft.irfft(pink_noise_fft, N)
    
    # Apply some scaling to the background noise (.5%)
    pink_noise = (pink_noise / np.std(pink_noise)) * 0.005
    
    # -------------------------------------------------------------------------
    # COMPOSITE OUTPUT
    # -------------------------------------------------------------------------
    # Merge submarine acoustic waves and background ambient noise into the final array
    composite_sea_signal = received_signal + pink_noise
    
    return composite_sea_signal
