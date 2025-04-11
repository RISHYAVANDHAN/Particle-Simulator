import numpy as np
from numpy.linalg import norm

class TwoPointInteraction:
    def __init__(self, particle1, particle2, force_law):
        """
        Initialize the interaction.

        Args:
            particle1 (Particle): The first particle in the interaction.
            particle2 (Particle): The second particle in the interaction.
            force_law (ForceLaw): The force law governing the interaction (e.g., SpringForceLaw).
        """
        self.particle1 = particle1
        self.particle2 = particle2
        self.force_law = force_law
        self.DOF = np.concatenate([particle1.DOF, particle2.DOF])

    def l(self, r):
        """
        Calculate the length of the spring.
        
        Args:
            r (np.ndarray): Array of positions
            
        Returns:
            float: The length of the spring
        """
        # Get positions of both particles
        r1 = self.particle1.slice(r)
        r2 = self.particle2.slice(r)
        r12 = r2 - r1
        return np.linalg.norm(r12)

    def n(self, r):
        """
        Calculate the unit vector along the spring.

        Args:
            r (np.ndarray): Array of positions with shape (n_particles, 3).

        Returns:
            np.ndarray: Unit vector along the spring.
        """
        r12 = self.particle2.slice(r) - self.particle1.slice(r)
        return r12 / norm(r12)

    def l_dot(self, r, v):
        """
        Calculate the rate of change of the spring length.
        
        Args:
            r (np.ndarray): Array of positions
            v (np.ndarray): Array of velocities
            
        Returns:
            float: Rate of change of the spring length
        """
        # Get velocities of both particles
        v1 = self.particle1.slice(v)
        v2 = self.particle2.slice(v)
        v12 = v2 - v1
        return np.dot(self.n(r), v12)

    def force(self, t, r, v):
        """
        Calculate the force exerted by the spring.

        Args:
            t (float): Current time.
            r (np.ndarray): Array of positions with shape (n_particles, 3).
            v (np.ndarray): Array of velocities with shape (n_particles, 3).

        Returns:
            tuple: Force vectors for the two particles (F1, F2).
        """
        l = self.l(r)
        l_dot = self.l_dot(r, v)
        la = self.force_law.la(t, l, l_dot)
        n = self.n(r)
        F1 = -la * n  # Force on particle 1
        F2 = la * n   # Force on particle 2
        return F1, F2


class System:
    def __init__(self, particles, interactions):
        """
        Initialize the system.

        Args:
            particles (list): List of Particle objects.
            interactions (list): List of TwoPointInteraction objects.
        """
        self.particles = particles
        self.interactions = interactions
        self.F0 = np.zeros((len(particles), 3))

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
            F1, F2 = interaction.force(t, r, v)
            F[self.particles.index(interaction.particle1)] += F1
            F[self.particles.index(interaction.particle2)] += F2
        return F
