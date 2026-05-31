import numpy as np

def calculate_submarine_trajectory(t, x0=-400, y0=120, z0=100, z_end=220):
    """
    Calculates the 3D trajectory of the target submarine over time and its
    absolute distance to the stationary mid-water hydrophone array.
    
    Parameters:
    -----------
    t : ndarray
        1D NumPy array representing the time steps of the simulation (seconds).
    x0, y0, z0 : float
        Initial 3D coordinates of the submarine at t = 0 (meters).
    z_end : float
        The final depth of the submarine at the end of the simulation (meters).
        
    Returns:
    --------
    coords : dict
        Dictionary containing arrays for 'x', 'y', 'z' positions of the sub.
    r_direct : ndarray
        1D array containing the true direct line-of-sight distance (meters) 
        from the submarine to the hydrophone at every time step.
    """
    # Total duration of the simulation based on the time array
    t_max = t[-1] if t[-1] > 0 else 1.0
    
    # 1. Define Constant Velocities (Diagonal horizontal traversal)
    v_x = 12.0  # m/s (~23.3 knots) traveling East
    v_y = -3.0  # m/s (~5.8 knots) drifting slightly North
    
    # 2. Compute Horizontal Positions over time
    x_sub = x0 + v_x * t
    y_sub = y0 + v_y * t
    
    # 3. Compute Vertical Position (Smooth, linear tactical deep dive)
    # Linearly interpolates depth from z0 to z_end over the simulation timeline
    z_sub = z0 + ((z_end - z0) / t_max) * t
    
    # 4. Define Hydrophone Coordinates (Stationary mid-water at origin)
    x_hydro = 0.0
    y_hydro = 0.0
    z_hydro = 200.0  # Suspended at 200m depth in a 400m water column
    
    # 5. Compute Euclidean Distance (Direct Path) using Vectorized Math
    r_direct = np.sqrt((x_sub - x_hydro)**2 + (y_sub - y_hydro)**2 + (z_sub - z_hydro)**2)
    
    # Store coordinates in a clean structure for subsequent reflection paths
    coords = {
        'x': x_sub,
        'y': y_sub,
        'z': z_sub
    }
    
    return coords, r_direct