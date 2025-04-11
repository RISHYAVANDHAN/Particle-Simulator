import numpy as np
import taichi as ti
from typing import Dict, Tuple
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from particle_simulator.Particle import Particle, System, TaichiSystem
from particle_simulator.Physics import Spring
from particle_simulator.Particle.interaction import TwoPointInteraction
from particle_simulator.Integrators import get_integrator
from particle_simulator.utils.conversions import extract_spring_lengths
from particle_simulator.Rendering.renderer import render_with_matplotlib, render_with_taichi

# Initialize Taichi backend
ti.init(arch=ti.gpu)  # Use ti.cpu if no GPU is available

def setup_reference_system() -> System:
    """Set up a basic triangle spring-mass system"""
    system = System()
    m = 1
    c = 1
    L = 1.0

    # Triangle configuration
    system.add_particles([
        Particle(m, r0=np.array([0, 0, 0])),
        Particle(m, r0=np.array([0, 1.1 * L, 0])),
        Particle(m, r0=np.array([1.2 * L, 0, 0]))
    ])

    system.add_interactions([
        TwoPointInteraction(system.particles[0], system.particles[1], Spring(c, L)),
        TwoPointInteraction(system.particles[1], system.particles[2], Spring(c, L)),
        TwoPointInteraction(system.particles[2], system.particles[0], Spring(c, L)),
    ])

    system.assemble()
    return system

def create_particle_animation(result):
    """Create animation of particle movement"""
    t = result[0]
    r = result[1]
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    particles, = ax.plot([], [], 'bo', markersize=8)

    def init():
        particles.set_data([], [])
        return particles,

    def update(frame):
        positions = r[frame].reshape(-1, 3)
        x = positions[:, 0]
        y = positions[:, 1]
        particles.set_data(x, y)
        return particles,

    ani = FuncAnimation(
        fig, update, frames=len(t),
        init_func=init, blit=True, interval=20
    )
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Three Particle Animation')
    plt.show()
    return ani

def visualize_spring_lengths(result: Dict) -> plt.Figure:
    """Plot spring lengths over time"""
    t = result[0]
    r = result[1]
    system = setup_reference_system()
    lengths = extract_spring_lengths(system, t, r)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, lengths[:, 0], label='$l_{12}$')
    ax.plot(t, lengths[:, 1], label='$l_{23}$')
    ax.plot(t, lengths[:, 2], label='$l_{31}$')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Length (m)')
    ax.set_title('Spring Lengths Over Time')
    ax.legend()
    ax.grid(True)
    plt.show()
    return fig

def run_simulation(method: str, use_taichi: bool = False):
    """Run the simulation and render results"""
    system = setup_reference_system()
    integrator = get_integrator(method, dt=1e-2, use_taichi=use_taichi)

    # Run the simulation
    result = integrator(system, h=1e-2, tf=3.0)

    # Call external renderer
    if use_taichi:
        render_with_taichi(result)
    else:
        render_with_matplotlib(result)
    
    visualize_spring_lengths(result)
    create_particle_animation(result)

    return result

if __name__ == "__main__":
    method = "explicit_euler"
    use_taichi = True  # Set False to run NumPy simulation

    run_simulation(method=method, use_taichi=use_taichi)
