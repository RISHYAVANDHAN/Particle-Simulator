from .eulers import ExplicitEulerIntegrator, SymplecticEulerIntegrator, ImplicitEulerIntegrator
from ..Particle.taichi_system import TaichiSystem
import numpy as np
import taichi as ti

def get_integrator(method="explicit_euler", dt=0.01, use_taichi=True, **kwargs):
    """
    Factory function to get the appropriate integrator.

    Args:
        method (str): Integration method to use
            - "explicit_euler": First-order explicit method
            - "symplectic_euler": First-order symplectic method
            - "implicit_euler": First-order implicit method
        dt (float): Time step size
        use_taichi (bool): Whether to use Taichi acceleration
        **kwargs: Additional arguments for specific integrators

    Returns:
        function or object: An integrator function (for Taichi) or an integrator instance (for CPU-based).
    """
    if not use_taichi:
        # Return CPU-based integrator
        integrators = {
            "explicit_euler": ExplicitEulerIntegrator(dt),
            "symplectic_euler": SymplecticEulerIntegrator(dt),
            "implicit_euler": ImplicitEulerIntegrator(dt, **kwargs),
        }

        if method in integrators:
            return integrators[method]
        else:
            print(f"Integration method '{method}' not recognized. Using explicit_euler.")
            return integrators["explicit_euler"]
    else:
        # Return Taichi-accelerated integration function
        return lambda system, h, tf: taichi_integration(system, h, tf, method)


def taichi_integration(system, dt, tf, method="explicit_euler"):
    t0 = system.t0
    n_steps = int((tf - t0) / dt) + 1

    # Taichi fields for particle properties
    num_particles = ti.field(dtype=int, shape=())
    num_particles[None] = len(system.particles)  # Set the number of particles as a Taichi field
    pos = ti.Vector.field(3, dtype=ti.f32, shape=num_particles[None])
    vel = ti.Vector.field(3, dtype=ti.f32, shape=num_particles[None])
    mass = ti.field(dtype=ti.f32, shape=num_particles[None])
    F = ti.Vector.field(3, dtype=ti.f32, shape=num_particles[None])

    # Initialize positions, velocities, and masses
    for i, p in enumerate(system.particles):
        pos[i] = ti.Vector(p.r0)
        vel[i] = ti.Vector(p.v0)
        mass[i] = p.mass

    @ti.kernel
    def update_positions_and_velocities(h: ti.f32):
        for i in range(num_particles[None]):  # Use num_particles field
            vel[i] += h * F[i] / mass[i]
            pos[i] += h * vel[i]
            
    # Run simulation and store results
    t = np.linspace(t0, tf, n_steps)
    r = np.zeros((n_steps, num_particles[None], 3), dtype=np.float32)
    v = np.zeros((n_steps, num_particles[None], 3), dtype=np.float32)

    for k in range(n_steps):
        # Save current positions and velocities
        for i in range(num_particles[None]):  # Use num_particles field
            r[k, i] = pos[i].to_numpy()
            v[k, i] = vel[i].to_numpy()

        # Compute forces in Python scope
        forces = system.Force(t0 + k * dt, r[k], v[k])

        # Ensure forces[i] is a 3-element iterable (list or np.array)
        for i in range(num_particles[None]):  # Use num_particles field
            if isinstance(forces[i], (list, np.ndarray)) and len(forces[i]) == 3:
                F[i] = ti.Vector([float(forces[i][0]), float(forces[i][1]), float(forces[i][2])])
            else:
                raise ValueError(f"Expected forces[{i}] to be a 3D vector-like object, but got: {forces[i]}")

        # Update positions and velocities
        update_positions_and_velocities(dt)

    return t, r, v



def to_taichi_system(system, n_steps, dt, integration_method="explicit_euler"):
    """
    Convert a CPU-based System to a TaichiSystem for acceleration.

    Args:
        system (System): The system to convert
        n_steps (int): Number of simulation steps
        dt (float): Time step size
        integration_method (str): Integration method to use

    Returns:
        TaichiSystem: Taichi-accelerated system
    """
    # Create a TaichiSystem
    ts = TaichiSystem(
        len(system.particles),
        len(system.interactions),
        n_steps,
        dt,
        integration_method,
    )

    # Copy particle data
    for i, p in enumerate(system.particles):
        ts.mass[i] = p.mass
        ts.pos[0, i] = np.array(p.r0, dtype=np.float32)
        ts.vel[0, i] = np.array(p.v0, dtype=np.float32)

    # Copy interaction data
    for i, interaction in enumerate(system.interactions):
        # Get particle indices
        p1_idx = system.particles.index(interaction.particle1)
        p2_idx = system.particles.index(interaction.particle2)

        # Set connection
        ts.interaction_connections[i, 0] = p1_idx
        ts.interaction_connections[i, 1] = p2_idx

        # Copy force law parameters element by element
        force_law = interaction.force_law
        parameters = force_law.get_parameters()  # Assuming this returns a list or array
        for j, param in enumerate(parameters):
            ts.force_law_params[i, j] = param  # Assign each parameter individually

    return ts
