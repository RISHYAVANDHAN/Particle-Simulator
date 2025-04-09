import numpy as np
from numpy.linalg import norm

class TwoPointInteraction:
    """
    Represents an interaction between two particles, governed by a force law.
    
    Attributes:
        particle1 (Particle): First particle in the interaction
        particle2 (Particle): Second particle in the interaction
        force_law (ForceLaw): Force law governing the interaction
        DOF (np.ndarray): Combined degrees of freedom for both particles
    """
    def __init__(self, particle1, particle2, force_law):
        """
        Initialize a two-point interaction between particles.
        
        Args:
            particle1 (Particle): First particle
            particle2 (Particle): Second particle
            force_law (ForceLaw): Force law governing the interaction
        """
        self.particle1 = particle1
        self.particle2 = particle2
        self.force_law = force_law
        # Combine DOFs from both particles
        self.DOF = None
        if particle1.DOF is not None and particle2.DOF is not None:
            self.DOF = np.concatenate([particle1.DOF, particle2.DOF])
    
    def l(self, r):
        """
        Calculate the distance between the two particles.
        
        Args:
            r (np.ndarray): Global position vector
            
        Returns:
            float: Distance between particles
        """
        r12 = self.particle2.slice(r) - self.particle1.slice(r)
        return norm(r12)
    
    def n(self, r):
        """
        Calculate the unit normal vector pointing from particle1 to particle2.
        
        Args:
            r (np.ndarray): Global position vector
            
        Returns:
            np.ndarray: Unit normal vector
        """
        r12 = self.particle2.slice(r) - self.particle1.slice(r)
        length = norm(r12)
        if length < 1e-10:  # Avoid division by zero
            return np.zeros(3)
        return r12 / length

    def l_dot(self, r, v):
        """
        Calculate the rate of change of distance between particles.
        
        Args:
            r (np.ndarray): Global position vector
            v (np.ndarray): Global velocity vector
            
        Returns:
            float: Rate of change of distance
        """
        v12 = self.particle2.slice(v) - self.particle1.slice(v)
        return np.dot(self.n(r), v12)

    def F(self, t, r, v):
        """
        Calculate the forces on both particles due to this interaction.
        
        Args:
            t (float): Current time
            r (np.ndarray): Global position vector
            v (np.ndarray): Global velocity vector
            
        Returns:
            np.ndarray: Force vector for both particles [F1, F2]
        """
        F = np.zeros(6)  # 3 components for each particle
        l = self.l(r)
        l_dot = self.l_dot(r, v)
        la = self.force_law.la(t, l, l_dot)
        n = self.n(r)
        
        # Apply forces in opposite directions
        F[:3] = -la * n  # Force on particle1
        F[3:] = la * n   # Force on particle2
        return F
