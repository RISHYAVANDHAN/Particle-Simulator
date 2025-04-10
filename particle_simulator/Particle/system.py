import numpy as np

class System:
    def __init__(self, t0=0, gravity=np.array([0, 0, -9.81])):
        self.particles = []
        self.interactions = []
        self.t0 = t0
        self.r0 = []
        self.v0 = []
        self.F0 = []
        self.m = []
        self.gravity = gravity
        self.last_particle_index = 0
        self.nDOF = 0  # Number of degrees of freedom

    def add_particles(self, particles):
        for p in particles:
            p.DOF = np.arange(3) + self.last_particle_index  # Assign 3 DOFs per particle
            self.last_particle_index += 3
            self.nDOF += 3
            self.particles.append(p)
            self.r0.extend(p.r0)
            self.v0.extend(p.v0)
            self.m.extend(p.mass * np.ones(3))
            self.F0.extend(p.mass * self.gravity)

    def add_particle(self, particle):
        if self._assembled:
            raise RuntimeError("Cannot add particles after assembly")
        particle.DOF = np.arange(3) + self.nDOF
        self.nDOF += 3
        self.particles.append(particle)

    def add_interactions(self, interactions):
        """
        Add interactions to the system.

        Args:
            interactions (list): List of `TwoPointInteraction` objects.
        """
        for interaction in interactions:
            self.interactions.append(interaction)

    def add_interaction(self, interaction):
        if self._assembled:
            raise RuntimeError("Cannot add interactions after assembly")
        self.interactions.append(interaction)

    def assemble(self):
        """Assemble the system's initial state."""
        self.r0 = np.array(self.r0)
        self.v0 = np.array(self.v0)
        self.m = np.array(self.m)
        self.F0 = np.array(self.F0)

    def Force(self, t, r, v):
        """Calculate forces on all particles."""
        F = np.zeros((len(self.particles), 3))  # Initialize force array

        for interaction in self.interactions:
            F1, F2 = interaction.force(t, r, v)
            idx1 = self.particles.index(interaction.particle1)
            idx2 = self.particles.index(interaction.particle2)

            F[idx1] += F1
            F[idx2] += F2

        # DEBUG: Print net force per particle
        for i, f in enumerate(F):
            print(f"Particle {i}: Force = {f}")

        return F  # Return the 3D force array directly
