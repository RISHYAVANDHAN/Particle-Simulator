# particle_simulator/utils/profiling.py
import time
import functools
import numpy as np
import taichi as ti
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Any, Tuple, Union

def profile_function(func: Callable) -> Callable:
    """
    A decorator to profile the execution time of a function.
    
    Args:
        func: The function to be profiled
        
    Returns:
        Callable: Wrapped function that prints execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.6f} seconds")
        return result
    return wrapper

def profile_simulation(system, integrators, h, tf, metrics):
    """
    Profile the simulation for different integrators.

    Args:
        system (System): The system to simulate.
        integrators (dict): Dictionary of integrators to profile.
        h (float): Time step size.
        tf (float): Final simulation time.
        metrics (list): List of metrics to compute.

    Returns:
        dict: Results for each integrator.
    """
    results = {}
    for name, integrator in integrators.items():
        print(f"Profiling {name}...")
        
        # Measure execution time
        start_time = time.time()
        if callable(integrator):
            t, r, v = integrator(system, h, tf)  # Call the function directly
        elif hasattr(integrator, "integrate"):
            t, r, v = integrator.integrate(system, h, tf)  # Call the `integrate` method
        else:
            raise TypeError(f"Integrator '{name}' is neither callable nor has an 'integrate' method.")
        end_time = time.time()
        
        # Store results
        results[name] = {
            "t": t,
            "r": r,
            "v": v,
            "time": end_time - start_time  # Store execution time
        }
        
        # Energy calculation if requested
        if "energy" in metrics:
            KE = np.zeros_like(t)  # Kinetic energy
            PE = np.zeros_like(t)  # Potential energy
            
            # Calculate energies
            for step in range(len(t)):
                # Calculate kinetic energy
                for i, particle in enumerate(system.particles):
                    vel = v[step, i*3:i*3+3]
                    KE[step] += 0.5 * particle.mass * np.sum(vel**2)
                
                # Calculate potential energy from interactions
                for interaction in system.interactions:
                    if hasattr(interaction.force_law, "potential_energy"):
                        ri = r[step, interaction.particle1.DOF]
                        rj = r[step, interaction.particle2.DOF]
                        PE[step] += interaction.force_law.potential_energy(ri, rj)
            
            # Total energy
            TE = KE + PE
            
            # Energy metrics
            results[name]["kinetic_energy"] = KE
            results[name]["potential_energy"] = PE
            results[name]["total_energy"] = TE
            results[name]["energy_drift"] = (TE[-1] - TE[0]) / (TE[0] if TE[0] != 0 else 1e-10)

    return results

def visualize_profiling_results(results: Dict[str, Any], 
                                 metrics: List[str] = ["time", "energy_drift", "energy"]) -> None:
    """
    Visualize the profiling results.
    
    Args:
        results: Results dictionary from profile_simulation
        metrics: Metrics to visualize
    """
    # Create a figure with subplots based on metrics
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 5*n_metrics))
    
    # If only one metric, wrap axes in a list
    if n_metrics == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        if metric == "time":
            # Bar chart for execution time
            names = list(results.keys())
            times = [results[name]["time"] for name in names]
            ax.bar(names, times)
            ax.set_ylabel("Execution Time (s)")
            ax.set_title("Integrator Performance Comparison")
            
        elif metric == "energy_drift":
            # Bar chart for energy drift
            names = list(results.keys())
            drifts = [results[name].get("energy_drift", 0) for name in names]
            ax.bar(names, drifts)
            ax.set_ylabel("Energy Drift (relative)")
            ax.set_title("Energy Conservation Comparison")
            
        elif metric == "energy":
            # Line chart for energy over time
            for name in results.keys():
                if "total_energy" in results[name]:
                    t = results[name]["t"]
                    E = results[name]["total_energy"]
                    E_normalized = E / E[0]  # Normalize to initial energy
                    ax.plot(t, E_normalized, label=f"{name}")
            
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Normalized Total Energy")
            ax.set_title("Energy Conservation Over Time")
            ax.legend()
            ax.grid(True)
            
    plt.tight_layout()
    return fig

@ti.kernel
def profile_taichi_kernel(n: ti.i32) -> ti.f32:
    """
    A simple kernel for profiling Taichi performance.
    
    Args:
        n: Number of iterations
        
    Returns:
        ti.f32: Time taken in seconds
    """
    t0 = ti.profiler.get_kernel_profiler_total_time()
    
    # Do some work
    sum_val = 0.0
    for i in range(n):
        sum_val += ti.sqrt(float(i))
    
    t1 = ti.profiler.get_kernel_profiler_total_time()
    return (t1 - t0) / 1000.0  # Convert to seconds

def taichi_performance_test(n_particles_list: List[int], n_steps: int = 100) -> Dict[str, List[float]]:
    """
    Test Taichi performance with different numbers of particles.
    
    Args:
        n_particles_list: List of particle counts to test
        n_steps: Number of simulation steps
        
    Returns:
        Dict: Performance metrics
    """
    ti.init(arch=ti.gpu, default_fp=ti.f32, kernel_profiler=True)
    
    results = {
        "n_particles": n_particles_list,
        "setup_time": [],
        "compute_time": [],
        "step_time": []
    }
    
    for n_particles in n_particles_list:
        print(f"Testing with {n_particles} particles...")
        
        # Measure setup time
        start_time = time.time()
        system = ti.field(ti.f32, shape=(n_particles, 3))
        velocities = ti.field(ti.f32, shape=(n_particles, 3))
        setup_time = time.time() - start_time
        results["setup_time"].append(setup_time)
        
        # Measure compute time
        @ti.kernel
        def compute():
            for i in range(n_particles):
                for j in range(i+1, n_particles):
                    # Simple distance calculation
                    dist = 0.0
                    for k in range(3):
                        dist += (system[i, k] - system[j, k])**2
                    dist = ti.sqrt(dist)
        
        ti.clear_kernel_profile_info()
        start_time = time.time()
        compute()
        compute_time = time.time() - start_time
        results["compute_time"].append(compute_time)
        
        # Measure step time
        @ti.kernel
        def step():
            for i in range(n_particles):
                for k in range(3):
                    system[i, k] += velocities[i, k] * 0.01
        
        ti.clear_kernel_profile_info()
        start_time = time.time()
        for _ in range(n_steps):
            step()
        step_time = time.time() - start_time
        results["step_time"].append(step_time / n_steps)
    
    return results

def plot_taichi_performance(results: Dict[str, List[float]]) -> plt.Figure:
    """
    Plot Taichi performance results.
    
    Args:
        results: Results from taichi_performance_test
        
    Returns:
        plt.Figure: The matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_particles = results["n_particles"]
    
    ax.plot(n_particles, results["setup_time"], 'o-', label="Setup Time")
    ax.plot(n_particles, results["compute_time"], 's-', label="Compute Time")
    ax.plot(n_particles, results["step_time"], '^-', label="Step Time")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Number of Particles")
    ax.set_ylabel("Time (s)")
    ax.set_title("Taichi Performance Scaling")
    ax.legend()
    ax.grid(True)
    
    return fig
