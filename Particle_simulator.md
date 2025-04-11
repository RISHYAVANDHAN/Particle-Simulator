# Particle-Simulator Documentation

## Overview

The Particle-Simulator is a physics-based simulation framework designed to model particle systems with various interactions, constraints, and numerical integration methods. It is built using the Taichi programming language for high-performance computation, enabling simulations to run efficiently on both CPUs and GPUs. The framework supports rendering using Matplotlib and Taichi GUI, and includes utilities for analyzing simulation results.

This documentation provides a detailed breakdown of the project, including its structure, modules, classes, functions, and usage.

## Project Structure

The project is organized into several directories and files, each serving a specific purpose. Below is the directory structure:

```
Particle-Simulator/
├── particle_simulator/
│   ├── Particle/
│   │   ├── particle.py
│   │   ├── system.py
│   │   ├── interaction.py
│   │   ├── taichi_system.py
│   ├── Physics/
│   │   ├── force_law.py
│   │   ├── constraints.py
│   ├── Integrators/
│   │   ├── base_integrators.py
│   │   ├── eulers.py
│   │   ├── factory.py
│   ├── Rendering/
│   │   ├── renderer.py
│   ├── utils/
│   │   ├── conversions.py
│   ├── simulation.py
├── README.md
├── LICENSE
├── setup.py
├── .gitignore
```

## Modules and Files

### 1. Particle

This module defines the core components of the particle simulation, including particles, interactions, and systems.

#### particle.py

Defines the `Particle` class, which represents a single particle in the simulation.

**Particle Class:**
- Represents a particle with mass, position, velocity, and degrees of freedom (DOF).
- **Attributes:**
  - `mass`: The mass of the particle.
  - `r0`: Initial position of the particle as a 3D NumPy array.
  - `v0`: Initial velocity of the particle as a 3D NumPy array.
  - `DOF`: Degrees of freedom indices in the global system (used for indexing global arrays).
- **Methods:**
  - `slice(arr)`: Extracts the 3D vector corresponding to the particle from a global array.
    - **Parameters:**
      - `arr`: A global array (e.g., position or velocity array).
    - **Returns:** A 3D vector corresponding to the particle.
  - `index`: Returns the particle's index in the global array.

#### system.py

Defines the `System` class, which represents a collection of particles and their interactions.

**System Class:**
- Represents a system of particles and their interactions.
- **Attributes:**
  - `particles`: A list of Particle objects in the system.
  - `interactions`: A list of interactions (e.g., springs) between particles.
  - `gravity`: A 3D vector representing the gravitational force applied to all particles.
- **Methods:**
  - `add_particles(particles)`: Adds multiple particles to the system.
    - **Parameters:**
      - `particles`: A list of Particle objects.
  - `add_interactions(interactions)`: Adds multiple interactions to the system.
    - **Parameters:**
      - `interactions`: A list of interaction objects (e.g., TwoPointInteraction).
  - `assemble()`: Assembles the system's initial state by computing global arrays for positions, velocities, and forces.
  - `Force(t, r, v)`: Computes the forces acting on all particles at a given time.
    - **Parameters:**
      - `t`: Current time.
      - `r`: Global position array.
      - `v`: Global velocity array.
    - **Returns:** A global force array.

#### interaction.py

Defines the `TwoPointInteraction` class, which models interactions (e.g., springs) between two particles.

**TwoPointInteraction Class:**
- Represents an interaction between two particles governed by a force law.
- **Attributes:**
  - `particle1`, `particle2`: The two particles involved in the interaction.
  - `force_law`: The force law governing the interaction (e.g., spring force).
- **Methods:**
  - `l(r)`: Computes the length of the interaction based on the global position array.
  - `n(r)`: Computes the unit vector along the interaction.
  - `l_dot(r, v)`: Computes the rate of change of the interaction length.
  - `force(t, r, v)`: Computes the forces exerted by the interaction.

#### taichi_system.py

Defines the `TaichiSystem` class, which accelerates the simulation using Taichi.

**TaichiSystem Class:**
- Represents a particle system implemented using Taichi for high-performance computation.
- **Attributes:**
  - `mass`, `pos`, `vel`: Taichi fields for particle properties.
  - `interaction_connections`: Taichi field for interaction connections.
  - `force_law_params`: Taichi field for force law parameters.
- **Methods:**
  - `initialize()`: Initializes forces and accelerations.
  - `compute_forces(step, t)`: Computes forces on all particles.
  - `explicit_euler_step(step)`: Performs an explicit Euler integration step.
  - `run_simulation()`: Runs the entire simulation.

### 2. Physics

This module defines the physics of the simulation, including force laws and constraints.

#### force_law.py

Defines various force laws for interactions between particles.

**ForceLaw Class:**
- Abstract base class for all force laws.
- **Methods:**
  - `la(t, l, l_dot)`: Computes the force magnitude based on length, velocity, and time.

**SpringForceLaw Class:**
- Models a linear spring force.
- **Methods:**
  - `la(t, l, l_dot)`: Computes the spring force.

**NonlinearSpringForceLaw Class:**
- Models a nonlinear spring force.
- **Methods:**
  - `la(t, l, l_dot)`: Computes the nonlinear spring force.

#### constraints.py

Defines constraints that restrict particle motion.

**Constraint Class:**
- Abstract base class for all constraints.
- **Methods:**
  - `apply(t, r, v)`: Applies the constraint to position and velocity vectors.

**FixedPointConstraint Class:**
- Fixes a particle at a specific position.

**PlaneConstraint Class:**
- Restricts particles to one side of a plane.

**SphereConstraint Class:**
- Restricts particles to remain within or on the surface of a sphere.

### 3. Integrators

This module provides numerical integrators for advancing the simulation in time.

#### base_integrators.py

Defines the base `Integrator` class.

**Integrator Class:**
- Abstract base class for all integrators.
- **Methods:**
  - `step(t, r, v, system)`: Advances the system by one time step.
  - `integrate(system, tf)`: Integrates the system from t0 to tf.

#### eulers.py

Implements Euler-based integration methods.

**ExplicitEulerIntegrator Class:**
- Implements the explicit Euler method.

**SymplecticEulerIntegrator Class:**
- Implements the symplectic Euler method.

**ImplicitEulerIntegrator Class:**
- Implements the implicit Euler method.

#### factory.py

Provides a factory function to select the appropriate integrator.

**get_integrator(method, dt, use_taichi, **kwargs):**
- Returns an integrator instance or function based on the specified method.

### 4. Rendering

This module provides rendering utilities for visualizing the simulation.

#### renderer.py

Defines functions for rendering the simulation.

**render_with_matplotlib(result):**
- Renders the simulation using Matplotlib animations.

**render_with_taichi(result):**
- Renders the simulation using Taichi's GUI.

### 5. utils

This module provides utility functions for conversions and data extraction.

#### conversions.py

Defines utility functions for working with simulation data.

**to_taichi_system(system):**
- Converts a `System` object to a `TaichiSystem`.

**extract_trajectory_for_particle(r, particle):**
- Extracts the trajectory of a specific particle.

**extract_spring_lengths(system, t, r):**
- Extracts the lengths of all springs over time.

### 6. simulation.py

This is the main script for running simulations.

**Functions:**
- `setup_reference_system()`: Sets up a basic spring-mass system.
- `create_particle_animation(result)`: Creates an animation of particle movement.
- `visualize_spring_lengths(result)`: Plots spring lengths over time.
- `run_simulation(method, use_taichi)`: Runs the simulation and renders results.

## Usage

### Run a Simulation:

1. Select an Integration Method: Modify the `method` variable in `simulation.py` to choose between:
   - `"explicit_euler"`
   - `"symplectic_euler"`
   - `"implicit_euler"`

### Visualize Results:

- Animations are displayed using Matplotlib or Taichi GUI.
- Spring lengths are plotted using `visualize_spring_lengths`.

## Dependencies

- taichi
- numpy
- matplotlib
- scipy

## License

This project is licensed under the MIT License. See the LICENSE file for details.