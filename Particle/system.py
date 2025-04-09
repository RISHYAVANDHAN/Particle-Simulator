import numpy as np
import taichi as ti

class System:
    """
    A physical system containing particles and their interactions.
    
    Attributes:
        particles (list): List of particles in the system
        interactions (list): List of interactions between particles
        t0 (float): Initial time
        r0 (np.ndarray): Initial position vector for all particles
        v0 (np.ndarray): Initial velocity vector for all particles
        F0 (np.ndarray): Initial force vector (e.g., gravity)
        m (np.ndarray): Mass vector for all particles
        gravity (np.ndarray): Gravity acceleration vector
        nDOF (int): Total number of degrees of freedom
    """
    def __init__(self, t0=0.0, gravity=np.array([0, 0, -9.81])):
        """
        Initialize a physical system.
        
        Args:
            t0 (float, optional): Initial time. Defaults to 0.0.
            gravity (np.ndarray, optional): Gravity acceleration vector. 
                                        Defaults to [0, 0, -9.81].
        """
        self.particles = []
        self.interactions = []
        self.t0 = t0
        self.r0 = []
        self.v0 = []
        self.F0 = []
        self.m = []
        self.gravity = np.array(gravity, dtype=np.float32)
        self.last_particle_index = 0
        self.nDOF = 0  # Number of degrees of freedom

    def add_particles(self, particles):
        """
        Add particles to the system.
        
        Args:
            particles (list): List of Particle objects to add
        """
        for p in particles:
            # Assign DOF indices
            p.DOF = np.arange(3) + self.last_particle_index
            self.last_particle_index += 3
            self.nDOF += 3
            
            # Store particle
            self.particles.append(p)
            
            # Store initial state
            self.r0.extend(p.r0)
            self.v0.extend(p.v0)
            
            # Store mass (repeated for each dimension)
            self.m.extend(p.mass * np.ones(3))
            
            # Initialize with gravity force
            self.F0.extend(p.mass * self.gravity)

    def add_interactions(self, interactions):
        """
        Add interactions to the system.
        
        Args:
            interactions (list): List of Interaction objects
        """
        for interaction in interactions:
            # Update DOF for the interaction if needed
            if interaction.DOF is None:
                interaction.DOF = np.concatenate([
                    interaction.particle1.DOF, 
                    interaction.particle2.DOF
                ])
            self.interactions.append(interaction)

    def assemble(self):
        """
        Assemble the system by converting lists to numpy arrays.
        This should be called after all particles and interactions are added.
        """
        self.r0 = np.array(self.r0, dtype=np.float32)
        self.v0 = np.array(self.v0, dtype=np.float32)
        self.m = np.array(self.m, dtype=np.float32)
        self.F0 = np.array(self.F0, dtype=np.float32)

    def F(self, t, r, v):
        """
        Compute the total force vector for all particles at time t.
        
        Args:
            t (float): Current time
            r (np.ndarray): Global position vector
            v (np.ndarray): Global velocity vector
            
        Returns:
            np.ndarray: Global force vector
        """
        # Start with constant forces (e.g., gravity)
        F = self.F0.copy()
        
        # Add forces from interactions
        for interaction in self.interactions:
            # Get force for this interaction
            F_interaction = interaction.F(t, r, v)
            
            # Add to global force vector
            F[interaction.DOF] += F_interaction
            
        return F
