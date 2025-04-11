import numpy as np
import taichi as ti
from ..Physics.force_law import SpringForceLaw

@ti.data_oriented
class TaichiSystem:
    """
    A Taichi-accelerated system for particle simulations.
    """

    def __init__(self, system, results, dt, integration_method="explicit_euler"):
        """
        Initialize the Taichi system using the original system and simulation results.

        Args:
            system (System): Original system containing particles and interactions
            results (Dict): Dictionary with simulation results
            dt (float): Time step size
            integration_method (str): Integration method to use
        """
        self.n_particles = len(system.particles)
        self.n_interactions = len(system.interactions)
        self.n_steps = len(results["t"])
        self.dt = dt
        self.integration_method = integration_method

        # Taichi fields
        self.mass = ti.field(ti.f32, shape=self.n_particles)
        self.pos = ti.Vector.field(3, ti.f32, shape=(self.n_steps, self.n_particles))
        self.vel = ti.Vector.field(3, ti.f32, shape=(self.n_steps, self.n_particles))
        self.force = ti.Vector.field(3, ti.f32, shape=self.n_particles)
        self.acc = ti.Vector.field(3, ti.f32, shape=self.n_particles)

        self.interaction_connections = ti.field(ti.i32, shape=(self.n_interactions, 2))
        self.force_law_params = ti.field(ti.f32, shape=(self.n_interactions, 5))

        # Gravity
        self.gravity = ti.Vector([0.0, 0.0, -9.81])

        # Load data from system and results
        self._initialize_from_data(system, results)

    def _initialize_from_data(self, system, results):
        """Populate Taichi fields with particle and interaction data."""
        # Set mass and initial positions/velocities
        for i, particle in enumerate(system.particles):
            self.mass[i] = particle.mass

        for step in range(self.n_steps):
            for i in range(self.n_particles):
                r = results["r"][step][i]
                v = results["v"][step][i]
                self.pos[step, i] = ti.Vector([r[0], r[1], r[2]])
                self.vel[step, i] = ti.Vector([v[0], v[1], v[2]])

        # Set interaction connections and force law parameters
        for i, interaction in enumerate(system.interactions):
            self.interaction_connections[i, 0] = system.particles.index(interaction.particle1)
            self.interaction_connections[i, 1] = system.particles.index(interaction.particle2)
            spring = interaction.force_law  # assumed to be Spring
            self.force_law_params[i, 0] = 0.0  # type 0 = spring
            self.force_law_params[i, 1] = spring.c
            self.force_law_params[i, 2] = spring.l0
            self.force_law_params[i, 3] = spring.damping if hasattr(spring, "damping") else 0.0
            self.force_law_params[i, 4] = 0.0  # unused

    @ti.kernel
    def initialize(self):
        for i in range(self.n_particles):
            self.force[i] = self.mass[i] * self.gravity
            self.acc[i] = self.force[i] / self.mass[i]

    @ti.func
    def compute_force(self, i: ti.i32, length: ti.f32, l_dot: ti.f32, t: ti.f32) -> ti.f32:
        force_type = int(self.force_law_params[i, 0])
        force_magnitude = 0.0
        if force_type == 0:
            stiffness = self.force_law_params[i, 1]
            rest_length = self.force_law_params[i, 2]
            damping = self.force_law_params[i, 3]
            force_magnitude = -stiffness * (length - rest_length) - damping * l_dot
        return force_magnitude

    @ti.kernel
    def compute_forces(self, step: ti.i32, t: ti.f32):
        for i in range(self.n_particles):
            self.force[i] = self.mass[i] * self.gravity

        for i in range(self.n_interactions):
            p1 = self.interaction_connections[i, 0]
            p2 = self.interaction_connections[i, 1]

            pos1 = self.pos[step, p1]
            pos2 = self.pos[step, p2]
            vel1 = self.vel[step, p1]
            vel2 = self.vel[step, p2]

            r12 = pos2 - pos1
            length = r12.norm()
            if length > 1e-10:
                direction = r12 / length
                l_dot = (vel2 - vel1).dot(direction)
                force_mag = self.compute_force(i, length, l_dot, t)
                force_vec = direction * force_mag
                self.force[p1] += force_vec
                self.force[p2] -= force_vec

        for i in range(self.n_particles):
            self.acc[i] = self.force[i] / self.mass[i]

    @ti.kernel
    def explicit_euler_step(self, step: ti.i32):
        for i in range(self.n_particles):
            self.vel[step+1, i] = self.vel[step, i] + self.dt * self.acc[i]
            self.pos[step+1, i] = self.pos[step, i] + self.dt * self.vel[step, i]

    @ti.kernel
    def symplectic_euler_step(self, step: ti.i32):
        for i in range(self.n_particles):
            self.vel[step+1, i] = self.vel[step, i] + self.dt * self.acc[i]
            self.pos[step+1, i] = self.pos[step, i] + self.dt * self.vel[step+1, i]

    @ti.kernel
    def implicit_euler_step_iteration(self, step: ti.i32, pos_new: ti.template(), vel_new: ti.template(), force_new: ti.template(), acc_new: ti.template()):
        for i in range(self.n_particles):
            vel_new[i] = self.vel[step, i] + self.dt * self.acc[i]
            pos_new[i] = self.pos[step, i] + self.dt * vel_new[i]

        for iter in range(5):
            for i in range(self.n_particles):
                force_new[i] = self.mass[i] * self.gravity

            for i in range(self.n_interactions):
                p1 = self.interaction_connections[i, 0]
                p2 = self.interaction_connections[i, 1]
                r12 = pos_new[p2] - pos_new[p1]
                length = r12.norm()
                direction = r12 / length
                v12 = vel_new[p2] - vel_new[p1]
                l_dot = v12.dot(direction)
                force_mag = self.compute_force(i, length, l_dot, self.dt * (step + 1))
                force_vec = direction * force_mag
                force_new[p1] += force_vec
                force_new[p2] -= force_vec

            for i in range(self.n_particles):
                acc_new[i] = force_new[i] / self.mass[i]
                vel_new[i] = self.vel[step, i] + self.dt * acc_new[i]
                pos_new[i] = self.pos[step, i] + self.dt * vel_new[i]

    def implicit_euler_step(self, step):
        pos_new = ti.Vector.field(3, ti.f32, shape=self.n_particles)
        vel_new = ti.Vector.field(3, ti.f32, shape=self.n_particles)
        force_new = ti.Vector.field(3, ti.f32, shape=self.n_particles)
        acc_new = ti.Vector.field(3, ti.f32, shape=self.n_particles)

        self.implicit_euler_step_iteration(step, pos_new, vel_new, force_new, acc_new)

        for i in range(self.n_particles):
            self.pos[step+1, i] = pos_new[i]
            self.vel[step+1, i] = vel_new[i]

    def run_simulation(self):
        """Run the complete simulation."""
        self.initialize()
        for step in range(self.n_steps - 1):
            t = step * self.dt
            self.compute_forces(step, t)
            if self.integration_method == "explicit_euler":
                self.explicit_euler_step(step)
            elif self.integration_method == "symplectic_euler":
                self.symplectic_euler_step(step)
            elif self.integration_method == "implicit_euler":
                self.implicit_euler_step(step)
            else:
                self.explicit_euler_step(step)
