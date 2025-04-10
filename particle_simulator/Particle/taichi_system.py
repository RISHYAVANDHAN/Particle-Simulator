import numpy as np
import taichi as ti
from ..Physics.force_law import SpringForceLaw

@ti.data_oriented
class TaichiSystem:
    """
    A Taichi-accelerated system for particle simulations.
    
    This class provides GPU acceleration for particle simulations
    using the Taichi programming language.
    """
    
    def __init__(self, n_particles, n_interactions, n_steps, dt, integration_method="explicit_euler"):
        """
        Initialize a Taichi-accelerated system.
        
        Args:
            n_particles (int): Number of particles
            n_interactions (int): Number of interactions
            n_steps (int): Number of simulation steps
            dt (float): Time step size
            integration_method (str): Integration method to use
        """
        # System parameters
        self.n_particles = n_particles
        self.n_interactions = n_interactions
        self.n_steps = n_steps
        self.dt = dt
        self.integration_method = integration_method
        
        # Particle data
        self.mass = ti.field(ti.f32, shape=n_particles)
        self.pos = ti.Vector.field(3, ti.f32, shape=(n_steps, n_particles))
        self.vel = ti.Vector.field(3, ti.f32, shape=(n_steps, n_particles))
        self.force = ti.Vector.field(3, ti.f32, shape=n_particles)
        self.acc = ti.Vector.field(3, ti.f32, shape=n_particles)
        
        # Interaction data
        self.interaction_connections = ti.field(ti.i32, shape=(n_interactions, 2))
        
        # Force law parameters (generic enough for various force laws)
        # [type, param1, param2, param3, param4]
        # For springs: [0, stiffness, rest_length, damping, 0]
        self.force_law_params = ti.field(ti.f32, shape=(n_interactions, 5))
        
        # For results tracking
        self.interaction_lengths = ti.field(ti.f32, shape=(n_steps, n_interactions))
        
        # Gravity
        self.gravity = ti.Vector([0.0, 0.0, -9.81])

    @ti.kernel
    def initialize(self):
        """Reset forces and initialize accelerations."""
        for i in range(self.n_particles):
            self.force[i] = self.mass[i] * self.gravity
            self.acc[i] = self.force[i] / self.mass[i]

    @ti.func
    def compute_force(self, i: ti.i32, length: ti.f32, l_dot: ti.f32, t: ti.f32) -> ti.f32:
        """
        Compute force based on force law parameters.
        
        Args:
            i (int): Interaction index
            length (float): Current length
            l_dot (float): Rate of change of length
            t (float): Current time
            
        Returns:
            float: Force magnitude
        """
        force_type = int(self.force_law_params[i, 0])
        force_magnitude = 0.0
        
        # Spring force law (type 0)
        if force_type == 0:
            stiffness = self.force_law_params[i, 1]
            rest_length = self.force_law_params[i, 2]
            damping = self.force_law_params[i, 3]
            
            # Hooke's law with optional damping
            force_magnitude = -stiffness * (length - rest_length) - damping * l_dot
            
        # Other force laws can be added here
        
        return force_magnitude

    @ti.kernel
    def compute_forces(self, step: ti.i32, t: ti.f32):
        """
        Compute all forces in the system.
        
        Args:
            step (int): Current simulation step
            t (float): Current simulation time
        """
        # Reset forces to gravity
        for i in range(self.n_particles):
            self.force[i] = self.mass[i] * self.gravity
        
        # Compute interaction forces
        for i in range(self.n_interactions):
            p1 = self.interaction_connections[i, 0]
            p2 = self.interaction_connections[i, 1]
            
            # Get positions and velocities
            pos1 = self.pos[step, p1]
            pos2 = self.pos[step, p2]
            vel1 = self.vel[step, p1]
            vel2 = self.vel[step, p2]
            
            # Calculate vector and length
            r12 = pos2 - pos1
            length = r12.norm()
            
            # Store the current length
            self.interaction_lengths[step, i] = length
            
            # Calculate the direction
            direction = r12.normalized()
            
            # Calculate the rate of change
            v12 = vel2 - vel1
            l_dot = v12.dot(direction)
            
            # Compute force based on the force law
            force_magnitude = self.compute_force(i, length, l_dot, t)
            
            # Apply forces
            force_vector = direction * force_magnitude
            self.force[p1] += force_vector
            self.force[p2] -= force_vector
        
        # Update accelerations
        for i in range(self.n_particles):
            self.acc[i] = self.force[i] / self.mass[i]

    @ti.kernel
    def explicit_euler_step(self, step: ti.i32):
        """
        Perform one step of explicit Euler integration.
        
        Args:
            step (int): Current simulation step
        """
        for i in range(self.n_particles):
            # Update velocity first (v += a*dt)
            self.vel[step+1, i] = self.vel[step, i] + self.dt * self.acc[i]
            
            # Then update position (x += v*dt)
            self.pos[step+1, i] = self.pos[step, i] + self.dt * self.vel[step, i]

    @ti.kernel
    def symplectic_euler_step(self, step: ti.i32):
        """
        Perform one step of symplectic Euler integration.
        
        Args:
            step (int): Current simulation step
        """
        for i in range(self.n_particles):
            # Update velocity first (v += a*dt)
            self.vel[step+1, i] = self.vel[step, i] + self.dt * self.acc[i]
            
            # Then update position using updated velocity (x += v_new*dt)
            self.pos[step+1, i] = self.pos[step, i] + self.dt * self.vel[step+1, i]

    @ti.kernel
    def implicit_euler_step_iteration(self, step: ti.i32, 
                                     pos_new: ti.template(), 
                                     vel_new: ti.template(),
                                     force_new: ti.template(),
                                     acc_new: ti.template()):
        """
        Perform one iteration of implicit Euler method.
        
        Args:
            step (int): Current simulation step
            pos_new (ti.Vector.field): New position field
            vel_new (ti.Vector.field): New velocity field
            force_new (ti.Vector.field): New force field
            acc_new (ti.Vector.field): New acceleration field
        """
        # This is a simplified implementation of implicit Euler
        # In a real implementation, you would need a proper nonlinear solver
        
        # Start with explicit Euler as initial guess
        for i in range(self.n_particles):
            vel_new[i] = self.vel[step, i] + self.dt * self.acc[i]
            pos_new[i] = self.pos[step, i] + self.dt * vel_new[i]
            
        # Simple fixed-point iteration for implicit Euler
        for iter in range(5):  # Limited number of iterations
            # Reset forces
            for i in range(self.n_particles):
                force_new[i] = self.mass[i] * self.gravity
            
            # Compute forces at the new state
            for i in range(self.n_interactions):
                p1 = self.interaction_connections[i, 0]
                p2 = self.interaction_connections[i, 1]
                
                r12 = pos_new[p2] - pos_new[p1]
                length = r12.norm()
                direction = r12.normalized()
                
                v12 = vel_new[p2] - vel_new[p1]
                l_dot = v12.dot(direction)
                
                # Get force at new state
                force_magnitude = self.compute_force(i, length, l_dot, self.dt * (step + 1))
                
                force_vector = direction * force_magnitude
                force_new[p1] += force_vector
                force_new[p2] -= force_vector
            
            # Compute new accelerations
            for i in range(self.n_particles):
                acc_new[i] = force_new[i] / self.mass[i]
            
            # Update velocity and position estimates
            for i in range(self.n_particles):
                vel_new[i] = self.vel[step, i] + self.dt * acc_new[i]
                pos_new[i] = self.pos[step, i] + self.dt * vel_new[i]

    def implicit_euler_step(self, step):
        """
        Perform one complete step of implicit Euler integration.
        
        Args:
            step (int): Current simulation step
        """
        # Create temporary fields for implicit Euler
        pos_new = ti.Vector.field(3, ti.f32, shape=self.n_particles)
        vel_new = ti.Vector.field(3, ti.f32, shape=self.n_particles)
        force_new = ti.Vector.field(3, ti.f32, shape=self.n_particles)
        acc_new = ti.Vector.field(3, ti.f32, shape=self.n_particles)
        
        # Perform implicit integration
        self.implicit_euler_step_iteration(step, pos_new, vel_new, force_new, acc_new)
        
        # Copy results
        for i in range(self.n_particles):
            self.pos[step+1, i] = pos_new[i]
            self.vel[step+1, i] = vel_new[i]

    def run_simulation(self):
        """Run the complete simulation using the specified integration method."""
        # Initialize the simulation
        self.initialize()
        
        # Run time steps
        for step in range(self.n_steps - 1):
            # Current time
            t = step * self.dt
            
            # Compute forces
            self.compute_forces(step, t)
            
            # Perform integration step
            if self.integration_method == "explicit_euler":
                self.explicit_euler_step(step)
            elif self.integration_method == "symplectic_euler":
                self.symplectic_euler_step(step)
            elif self.integration_method == "implicit_euler":
                self.implicit_euler_step(step)
            else:
                # Default to explicit Euler
                self.explicit_euler_step(step)
