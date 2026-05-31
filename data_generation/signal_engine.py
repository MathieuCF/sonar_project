import numpy as np
from scipy.signal import lfilter

def generate_sonar_signal(t, coords, r_direct, fs=1000.0):
    """
    Synthesizes a high-fidelity passive sonar time-series recording featuring 
    a moving submarine signature distorted by Arctic multi-path propagation.
    """
    N = len(t)
    c = 1440.0  # Speed of sound in cold Arctic water (m/s)
    
    # -------------------------------------------------------------------------
    # PART 1: SYNTHESIZE RAW SUBMARINE SIGNATURE (Machinery + Cavitation)
    # -------------------------------------------------------------------------
    # Narrowband machinery tones (80 Hz fundamental + 160 Hz harmonic)
    machinery_tone = np.sin(2 * np.pi * 80.0 * t) + 0.4 * np.sin(2 * np.pi * 160.0 * t)
    
    # Broadband propeller cavitation (White noise shaped into broadband hiss)
    raw_hiss = np.random.normal(0, 1, N)
    # Simple low-pass filter to shape the broadband cavitation curve
    b, a = [0.1], [1, -0.9] 
    shaped_hiss = lfilter(b, a, raw_hiss)
    
    # Shaft/Blade modulation (Trevorrow Effect: Hiss amplitude pulses at 1.5 Hz)
    blade_rate = 1.5  # Hz
    modulation = 1.0 + 0.6 * np.sin(2 * np.pi * blade_rate * t)
    cavitation_signature = shaped_hiss * modulation
    
    # Combine into a single core structural signature emitted by the boat
    raw_sub_signal = 1 * machinery_tone + 0.6 * cavitation_signature
    
    # -------------------------------------------------------------------------
    # PART 2: MULTI-PATH ACOUSTIC PROPAGATION (The Arctic Waveguide)
    # -------------------------------------------------------------------------
    # Hydrophone depth is fixed at 200m in a 400m water column
    z_hydro = 200.0
    H = 400.0
    
    # Initialize our final hydrophone receiver array
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
        
        # Constructive accumulation with environmental damping (attenuation)
        # We ensure indices stay within the boundaries of our simulated time array
        if idx_d >= 0:
            received_signal[i] += raw_sub_signal[idx_d] / r_d
        if idx_ice >= 0:
            received_signal[i] += (-0.4 * raw_sub_signal[idx_ice]) / r_ice
        if idx_floor >= 0:
            received_signal[i] += (0.6 * raw_sub_signal[idx_floor]) / r_floor
            
    return received_signal