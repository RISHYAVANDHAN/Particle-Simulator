import numpy as np
import taichi as ti
from Physics import SpringForceLaw

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

def calculate_system_energy(system, t, r, v):
    """
    Calculate kinetic, potential, and total energy of the system over time.
    
    Args:
        system (System): The system being simulated
        t (np.ndarray): Time array
        r (np.ndarray): Position array
        v (np.ndarray): Velocity array
        
    Returns:
        tuple: (KE, PE, E) arrays for kinetic, potential, and total energy
    """
    n_steps = len(t)
    
    # Initialize energy arrays
    KE = np.zeros(n_steps)  # Kinetic energy
    PE = np.zeros(n_steps)  # Potential energy
    
    for step in range(n_steps):
        # Calculate kinetic energy
        for i, particle in enumerate(system.particles):
            vel = v[step, particle.DOF]
            KE[step] += 0.5 * particle.mass * np.dot(vel, vel)
        
        # Calculate potential energy
        # 1. Gravitational potential energy
        for i, particle in enumerate(system.particles):
            pos = r[step, particle.DOF]
            # Height from reference level (assuming gravity is in -z direction)
            height = pos[2]  # z-coordinate
            PE[step] += particle.mass * -system.gravity[2] * height
        
        # 2. Spring potential energy
        for interaction in system.interactions:
            force_law = interaction.force_law
            if isinstance(force_law, SpringForceLaw):
                length = interaction.l(r[step])
                PE[step] += 0.5 * force_law.k * (length - force_law.l0) ** 2
    
    # Total energy
    E = KE + PE
    
    return KE, PE, E
