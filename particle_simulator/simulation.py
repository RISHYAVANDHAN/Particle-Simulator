import numpy as np
import taichi as ti
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from particle_simulator.Particle import Particle, System, TaichiSystem
from particle_simulator.Physics import SpringForceLaw, Spring
from particle_simulator.Particle.interaction import TwoPointInteraction
from particle_simulator.Integrators import get_integrator
from particle_simulator.utils.profiling import profile_simulation, visualize_profiling_results
from particle_simulator.utils.conversions import extract_spring_lengths, calculate_system_energy

ti.init(arch=ti.gpu)  # use ti.cpu if you have no GPU

def setup_reference_system() -> System:
    system = System()
    m = 1
    c = 1
    L = 1.0

    # Triangle of particles
    system.add_particles([
        Particle(m, r0=np.zeros(3)),
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

def run_simulation(method: str, use_taichi: bool = False) -> Tuple[Dict, plt.Figure]:
    """Run and profile simulation"""
    system = setup_reference_system()
    integrator = get_integrator(method, dt=1e-2, use_taichi=use_taichi)

    results = profile_simulation(
        system=system,
        integrators={method: integrator},
        h=1e-2,
        tf=3.0,
        metrics=["time", "energy"]
    )

    fig = visualize_profiling_results(results, metrics=["time", "energy_drift", "energy"])
    visualize_spring_lengths(results[method])
    create_particle_animation(results[method])

    return results, fig

def visualize_spring_lengths(result: Dict) -> plt.Figure:
    """Plot spring lengths over time"""
    t = result["t"]
    r = result["r"]
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

def create_particle_animation(result: Dict) -> FuncAnimation:
    """Create animation of particle movement"""
    t = result["t"]
    r = result["r"]
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    particles, = ax.plot([], [], 'bo', markersize=8)

    def init():
        particles.set_data([], [])
        return particles,

    def update(frame):
        x = r[frame, 0::3]
        y = r[frame, 1::3]
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

if __name__ == "__main__":
    method = "explicit_euler"

    # Run with Taichi
    taichi_results, _ = run_simulation(method=method, use_taichi=True)

    # Run with NumPy
    #numpy_results, _ = run_simulation(method=method, use_taichi=False)
