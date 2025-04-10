import numpy as np
import taichi as ti

class Particle:
    """
    A particle in the simulation with mass, position, and velocity.
    
    Attributes:
        mass (float): Mass of the particle
        r0 (np.ndarray): Initial position vector (3D)
        v0 (np.ndarray): Initial velocity vector (3D)
        DOF (np.ndarray): Degrees of freedom indices in the global system
    """
    def __init__(self, mass, r0=np.zeros(3), v0=np.zeros(3)):
        """
        Initialize a particle with mass, position, and velocity.
        
        Args:
            mass (float): Mass of the particle
            r0 (np.ndarray, optional): Initial position vector. Defaults to [0, 0, 0].
            v0 (np.ndarray, optional): Initial velocity vector. Defaults to [0, 0, 0].
        """
        self.mass = mass
        self.r0 = r0
        self.v0 = v0
        self.DOF = None  # Will be assigned when added to a system

    @property
    def index(self):
        """
        Returns the particle index in the global array from the first DOF entry.
        Assumes DOF = [3*i, 3*i+1, 3*i+2]
        """
        return self.DOF[0] // 3

    def slice(self, arr):
        """
        Extract the 3D vector corresponding to this particle from a global array.

        Args:
            arr (np.ndarray): Array of positions or velocities with shape (n_particles, 3).

        Returns:
            np.ndarray: The 3D vector (position or velocity) of this particle.
        """
        return arr[self.index]

    def F(self, t, r, v):
        """
        Compute the global force vector for the system.

        Args:
            t (float): Current time.
            r (np.ndarray): Array of positions with shape (n_particles, 3).
            v (np.ndarray): Array of velocities with shape (n_particles, 3).

        Returns:
            np.ndarray: Global force vector with shape (n_particles, 3).
        """
        F = self.F0.copy()
        for interaction in self.interactions:
            if interaction.particle1 not in self.particles:
                raise ValueError(f"Particle1 {interaction.particle1} not found in system particles.")
            if interaction.particle2 not in self.particles:
                raise ValueError(f"Particle2 {interaction.particle2} not found in system particles.")
            
            F1, F2 = interaction.F(t, r, v)

            assert F1.shape == (3,), f"F1 shape mismatch: got {F1.shape}, expected (3,)"
            assert F2.shape == (3,), f"F2 shape mismatch: got {F2.shape}, expected (3,)"

            idx1 = self.particles.index(interaction.particle1)
            idx2 = self.particles.index(interaction.particle2)

            F[idx1] += F1
            F[idx2] += F2

        return F
