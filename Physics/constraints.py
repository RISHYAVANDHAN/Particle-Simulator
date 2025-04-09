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
        normal (np.ndarray): Normal vector to the plane
        point (np.ndarray): A point on the plane
        particles (list): List of particles to constrain
        restitution (float): Coefficient of restitution for bounces
    """
    def __init__(self, normal, point, particles, restitution=0.8):
        """
        Initialize a plane constraint.
        
        Args:
            normal (np.ndarray): Normal vector to the plane
            point (np.ndarray): A point on the plane
            particles (list): List of particles to constrain
            restitution (float, optional): Coefficient of restitution. Defaults to 0.8.
        """
        self.normal = np.array(normal, dtype=np.float32)
        self.normal = self.normal / np.linalg.norm(self.normal)  # Normalize
        self.point = np.array(point, dtype=np.float32)
        self.particles = particles
        self.restitution = restitution
    
    def apply(self, t, r, v):
        """
        Apply plane constraint.
        
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
        
        for particle in self.particles:
            # Get particle position
            pos = r[particle.DOF]
            vel = v[particle.DOF]
            
            # Calculate signed distance to plane
            vec_to_plane = pos - self.point
            distance = np.dot(vec_to_plane, self.normal)
            
            # If particle is on wrong side of plane
            if distance < 0:
                # Project back to plane
                r_new[particle.DOF] = pos - distance * self.normal
                
                # Reflect velocity with restitution
                normal_vel = np.dot(vel, self.normal)
                if normal_vel < 0:  # Only reflect if moving toward the plane
                    v_new[particle.DOF] = vel - (1 + self.restitution) * normal_vel * self.normal
        
        return r_new, v_new
