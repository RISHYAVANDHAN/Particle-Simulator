# particle_simulator/main.py
import numpy as np
import taichi as ti
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

# Import from your modules
from particle_simulator.Particle import Particle, System, TaichiSystem
from particle_simulator.Physics import SpringForceLaw
from particle_simulator.Particle.interaction import TwoPointInteraction
from particle_simulator.Integrators import get_integrator
from particle_simulator.utils.profiling import profile_simulation, visualize_profiling_results
from particle_simulator.utils.conversions import extract_spring_lengths, calculate_system_energy

def setup_reference_system() -> System:
    """Create the same 3-particle system as your NumPy version"""
    system = System(gravity=np.array([0, 0, -9.81]))
    
    # Create particles (same configuration as NumPy version)
    m = 1.0
    L = 1.0
    c = 1.0
    
    particles = [
        Particle(mass=m, r0=np.zeros(3)),
        Particle(mass=m, r0=np.array([0, 1.1*L, 0])),
        Particle(mass=m, r0=np.array([1.2*L, 0, 0]))
    ]
    system.add_particles(particles)
    
    # Create springs between particles
    spring_law = SpringForceLaw(stiffness=c, rest_length=L)
    interactions = [
        TwoPointInteraction(particles[0], particles[1], spring_law),
        TwoPointInteraction(particles[1], particles[2], spring_law),
        TwoPointInteraction(particles[2], particles[0], spring_law)
    ]
    system.add_interactions(interactions)
    
    system.assemble()
    return system

def run_simulation(use_taichi: bool = False) -> Tuple[Dict, plt.Figure]:
    """Run simulation with profiling and visualization"""
    system = setup_reference_system()
    
    # Define integrators to test
    integrators = {
        "explicit_euler": get_integrator("explicit_euler", dt=1e-2, use_taichi=use_taichi),
        "symplectic_euler": get_integrator("symplectic_euler", dt=1e-2, use_taichi=use_taichi)
    }
    
    # Profile the simulation
    results = profile_simulation(
        system=system,
        integrators=integrators,
        h=1e-2,
        tf=3.0,
        metrics=["time", "energy"]
    )
    
    # Visualize results
    fig = visualize_profiling_results(results, metrics=["time", "energy_drift", "energy"])
    
    # Additional visualization matching NumPy version
    visualize_spring_lengths(results["explicit_euler"])
    create_particle_animation(results["explicit_euler"])
    
    return results, fig

def visualize_spring_lengths(result: Dict) -> plt.Figure:
    """Plot spring lengths over time (matches NumPy version)"""
    t = result["t"]
    r = result["r"]
    
    # Calculate spring lengths
    system = setup_reference_system()
    lengths = extract_spring_lengths(system, t, r)
    
    # Plot
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
    """Create 2D particle animation (matches NumPy version)"""
    t = result["t"]
    r = result["r"]
    
    # Setup figure
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    
    # Initialize plot elements
    particles, = ax.plot([], [], 'bo', markersize=8)
    
    def init():
        particles.set_data([], [])
        return particles,
    
    def update(frame):
        x = r[frame, 0::3]  # X positions for all particles
        y = r[frame, 1::3]  # Y positions for all particles
        particles.set_data(x, y)
        return particles,
    
    # Create animation
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
    # Run with Taichi acceleration
    taichi_results, _ = run_simulation(use_taichi=True)
    
    # For comparison, run without Taichi
    numpy_results, _ = run_simulation(use_taichi=False)