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
    def __init__(self, mass, r0=None, v0=None):
        """
        Initialize a particle with mass, position, and velocity.
        
        Args:
            mass (float): Mass of the particle
            r0 (np.ndarray, optional): Initial position vector. Defaults to [0,0,0].
            v0 (np.ndarray, optional): Initial velocity vector. Defaults to [0,0,0].
        """
        self.mass = mass
        self.r0 = np.zeros(3) if r0 is None else np.array(r0, dtype=np.float32)
        self.v0 = np.zeros(3) if v0 is None else np.array(v0, dtype=np.float32)
        self.DOF = None  # Will be assigned when added to a system

    def slice(self, r):
        """
        Extract the position of this particle from the global position vector.
        
        Args:
            r (np.ndarray): Global position vector
            
        Returns:
            np.ndarray: Position of this particle
        """
        if self.DOF is None:
            raise ValueError("Particle has not been added to a system")
        return r[self.DOF]
