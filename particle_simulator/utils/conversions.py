import numpy as np
import taichi as ti
from Physics import SpringForceLaw

def to_taichi_system(system):
    """
    Convert a System object to a Taichi-compatible system.

    Args:
        system (System): The system to convert.

    Returns:
        TaichiSystem: A Taichi-compatible version of the system.
    """
    from particle_simulator.Particle.taichi_system import TaichiSystem
    
    # Create a TaichiSystem from the given system
    taichi_system = TaichiSystem(
        particles=system.particles,
        interactions=system.interactions,
        gravity=system.gravity
    )
    return taichi_system

def extract_trajectory_for_particle(r, particle):
    """
    Extract trajectory for a specific particle from global position array.
    
    Args:
        r (np.ndarray): Global position array (n_steps x nDOF)
        particle (Particle): Particle to extract trajectory for
        
    Returns:
        np.ndarray: Trajectory for the particle (n_steps x 3)
    """
    if particle.DOF is None:
        raise ValueError("Particle has not been added to a system")
    
    return r[:, particle.DOF]

def extract_spring_lengths(system, t, r):
    """
    Extract lengths of all springs over time.
    
    Args:
        system (System): The system being simulated
        t (np.ndarray): Time array
        r (np.ndarray): Position array
        
    Returns:
        np.ndarray: Spring lengths (n_steps x n_springs)
    """
    n_steps = len(t)
    n_interactions = len(system.interactions)
    
    lengths = np.zeros((n_steps, n_interactions))
    
    for step in range(n_steps):
        for i, interaction in enumerate(system.interactions):
            lengths[step, i] = interaction.l(r[step])
    
    return lengths

