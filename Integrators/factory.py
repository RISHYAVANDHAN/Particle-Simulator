from .euler_integrator import ExplicitEulerIntegrator, SymplecticEulerIntegrator, ImplicitEulerIntegrator
from ..core.taichi_system import TaichiSystem
import numpy as np

def get_integrator(method="explicit_euler", dt=0.01, use_taichi=True, **kwargs):
    """
    Factory function to get the appropriate integrator.
    
    Args:
        method (str): Integration method to use
            - "explicit_euler": First-order explicit method
            - "symplectic_euler": First-order symplectic method
            - "implicit_euler": First-order implicit method
        dt (float): Time step size
        use_taichi (bool): Whether to use Taichi acceleration
        **kwargs: Additional arguments for specific integrators
            
    Returns:
        object: An integrator instance or function
    """
    if not use_taichi:
        # Return CPU-based integrator
        integrators = {
            "explicit_euler": ExplicitEulerIntegrator(dt),
            "symplectic_euler": SymplecticEulerIntegrator(dt), 
            "implicit_euler": ImplicitEulerIntegrator(dt, **kwargs),
        }
        
        if method in integrators:
            return integrators[method]
        else:
            print(f"Integration method '{method}' not recognized. Using explicit_euler.")
            return integrators["explicit_euler"]
    else:
        # Return Taichi-accelerated integration function
        return lambda system, tf: taichi_integration(system, dt, tf, method)

def taichi_integration(system, dt, tf, method="explicit_euler"):
    """
    Integrated simulation using Taichi with various integration methods.
    
    Args:
        system (System): The system to simulate
        dt (float): Time step size
        tf (float): Final simulation time
        method (str): Integration method to use
        
    Returns:
        tuple: (time array, position array, velocity array, taichi system, lengths)
    """
    t0 = system.t0
    n_steps = int((tf - t0) / dt) + 1
    
    # Validate integration method
    valid_methods = ["explicit_euler", "symplectic_euler", "implicit_euler"]
    if method not in valid_methods:
        print(f"Warning: Integration method '{method}' not implemented in Taichi. Using explicit_euler instead.")
        method = "explicit_euler"
    
    # Convert to Taichi system
    ts = to_taichi_system(system, n_steps, dt, method)
    
    # Run simulation
    ts.run_simulation()
    
    # Create time array
    t = np.linspace(t0, tf, n_steps)
    
    # Extract position and velocity data
    r = np.zeros((n_steps, system.nDOF))
    v = np.zeros((n_steps, system.nDOF))
    
    # Transfer data from Taichi fields to numpy arrays
    for step in range(n_steps):
        for i in range(len(system.particles)):
            pos = ts.pos[step, i].to_numpy()
            vel = ts.vel[step, i].to_numpy()
            
            r[step, i*3:i*3+3] = pos
            v[step, i*3:i*3+3] = vel
    
    # Extract interaction lengths if available
    lengths = np.zeros((n_steps, len(system.interactions)))
    for step in range(n_steps):
        for i in range(len(system.interactions)):
            lengths[step, i] = ts.interaction_lengths[step, i]
    
    return t, r, v, ts, lengths

def to_taichi_system(system, n_steps, dt, integration_method="explicit_euler"):
    """
    Convert a CPU-based System to a TaichiSystem for acceleration.
    
    Args:
        system (System): The system to convert
        n_steps (int): Number of simulation steps
        dt (float): Time step size
        integration_method (str): Integration method to use
        
    Returns:
        TaichiSystem: Taichi-accelerated system
    """
    # Create a TaichiSystem
    ts = TaichiSystem(
        len(system.particles), 
        len(system.interactions), 
        n_steps, 
        dt, 
        integration_method
    )
    
    # Copy particle data
    for i, p in enumerate(system.particles):
        ts.mass[i] = p.mass
        ts.pos[0, i] = np.array(p.r0, dtype=np.float32)
        ts.vel[0, i] = np.array(p.v0, dtype=np.float32)
    
    # Copy interaction data
    for i, interaction in enumerate(system.interactions):
        # Get particle indices
        p1_idx = system.particles.index(interaction.particle1)
        p2_idx = system.particles.index(interaction.particle2)
        
        # Set connection
        ts.interaction_connections[i, 0] = p1_idx
        ts.interaction_connections[i, 1] = p2_idx
        
        # Copy force law parameters
        force_law = interaction.force_law
        ts.force_law_params[i] = force_law.get_parameters()
    
    return ts
