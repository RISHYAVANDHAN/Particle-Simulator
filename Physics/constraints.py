import numpy as np

class Constraint:
    """
    Base class for constraints in the particle system.
    
    Constraints restrict the motion of particles in some way.
    """
    def apply(self, t, r, v):
        """
        Apply the constraint to position and velocity vectors.
        
        Args:
            t (float): Current time
            r (np.ndarray): Position vector to be modified
            v (np.ndarray): Velocity vector to be modified
            
        Returns:
            tuple: Modified position and velocity vectors
        """
        raise NotImplementedError("Constraint is an abstract class")

class FixedPointConstraint(Constraint):
    """
    Constraint that fixes a particle at a specific position.
    
    Attributes:
        particle (Particle): The particle to constrain
        fixed_position (np.ndarray): Position to fix the particle at
    """
    def __init__(self, particle, fixed_position=None):
        """
        Initialize a fixed point constraint.
        
        Args:
            particle (Particle): The particle to fix
            fixed_position (np.ndarray, optional): Position to fix at.
                                           Defaults to particle's initial position.
        """
        self.particle = particle 
        self.fixed_position = fixed_position if fixed_position is not None else particle.r0.copy()
    
    def apply(self, t, r, v):
        """
        Apply fixed position constraint.
        
        Args:
            t (float): Current time
            r (np.ndarray): Position vector to modify
            v (np.ndarray): Velocity vector to modify
            
        Returns:
            tuple: Modified position and velocity vectors
        """
        # Make copies to avoid modifying originals
        r_new = r.copy()
        v_new = v.copy()
        
        # Fix position
        r_new[self.particle.DOF] = self.fixed_position
        
        # Set velocity to zero
        v_new[self.particle.DOF] = 0.0
        
        return r_new, v_new

class PlaneConstraint(Constraint):
    """
    Constraint that restricts particles to one side of a plane.
    
    Attributes:
        normal (np.ndarray): Normal vector to the
