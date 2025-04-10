import numpy as np
from scipy.optimize import fsolve
from .base_integrators import Integrator

class ExplicitEulerIntegrator(Integrator):
    """
    Explicit Euler integration method.
    
    Updates velocities and positions using the current accelerations:
    v(t+dt) = v(t) + a(t) * dt
    r(t+dt) = r(t) + v(t) * dt
    """
    
    def __init__(self, dt):
        self.dt = dt

    def step(self, t, r, v, system):
        """
        Advance the system by one time step using explicit Euler.
        
        Args:
            t (float): Current time
            r (np.ndarray): Current position vector
            v (np.ndarray): Current velocity vector
            system (System): The system to simulate
            
        Returns:
            tuple: New time, position vector, and velocity vector
        """
        # Compute forces and accelerations
        F = system.F(t, r, v)
        a = F / system.m
        
        # Update velocity
        v_new = v + self.dt * a
        
        # Update position
        r_new = r + self.dt * v
        
        # Return new state
        return t + self.dt, r_new, v_new

    def integrate(self, system, h, tf):
        """
        Perform explicit Euler integration.

        Args:
            system (System): The system to integrate.
            h (float): Time step size.
            tf (float): Final simulation time.

        Returns:
            tuple: (time array, position array, velocity array)
        """
        t0 = system.t0
        n_steps = int((tf - t0) / h) + 1

        t = np.linspace(t0, tf, n_steps)
        r = np.zeros((n_steps, system.nDOF))
        v = np.zeros((n_steps, system.nDOF))

        # Initialize positions and velocities
        r[0] = system.r0
        v[0] = system.v0

        # Perform integration
        for step in range(1, n_steps):
            F = system.F(t[step - 1], r[step - 1], v[step - 1])
            v[step] = v[step - 1] + h * F / system.m
            r[step] = r[step - 1] + h * v[step - 1]

        return t, r, v

class SymplecticEulerIntegrator(Integrator):
    """
    Symplectic Euler integration method.
    
    First updates velocities, then uses the updated velocities to update positions:
    v(t+dt) = v(t) + a(t) * dt
    r(t+dt) = r(t) + v(t+dt) * dt
    
    This method conserves energy better than explicit Euler for oscillatory systems.
    """
    
    def step(self, t, r, v, system):
        """
        Advance the system by one time step using symplectic Euler.
        
        Args:
            t (float): Current time
            r (np.ndarray): Current position vector
            v (np.ndarray): Current velocity vector
            system (System): The system to simulate
            
        Returns:
            tuple: New time, position vector, and velocity vector
        """
        # Compute forces and accelerations
        F = system.F(t, r, v)
        a = F / system.m
        
        # Update velocity first
        v_new = v + self.dt * a
        
        # Update position using the new velocity
        r_new = r + self.dt * v_new
        
        # Return new state
        return t + self.dt, r_new, v_new

class ImplicitEulerIntegrator(Integrator):
    """
    Implicit Euler integration method.
    
    Solves for the state at the next time step by considering the forces at that future time:
    v(t+dt) = v(t) + a(t+dt) * dt
    r(t+dt) = r(t) + v(t+dt) * dt
    
    This method is unconditionally stable but requires solving a nonlinear system.
    """
    
    def __init__(self, dt, tol=1e-6, max_iter=100):
        """
        Initialize an implicit Euler integrator.
        
        Args:
            dt (float): Time step size
            tol (float, optional): Tolerance for convergence. Defaults to 1e-6.
            max_iter (int, optional): Maximum iterations. Defaults to 100.
        """
        super().__init__(dt)
        self.tol = tol
        self.max_iter = max_iter
    
    def step(self, t, r, v, system):
        """
        Advance the system by one time step using implicit Euler.
        
        Args:
            t (float): Current time
            r (np.ndarray): Current position vector
            v (np.ndarray): Current velocity vector
            system (System): The system to simulate
            
        Returns:
            tuple: New time, position vector, and velocity vector
        """
        t_new = t + self.dt
        
        # Define the residual function for the nonlinear system
        def residual(x):
            r_new = x[:system.nDOF]
            v_new = x[system.nDOF:]
            
            # Compute forces at the new state
            F_new = system.F(t_new, r_new, v_new)
            a_new = F_new / system.m
            
            # Residuals for velocity and position updates
            res_v = v_new - v - self.dt * a_new
            res_r = r_new - r - self.dt * v_new
            
            return np.concatenate([res_r, res_v])
        
        # Initial guess from explicit Euler
        F_current = system.F(t, r, v)
        a_current = F_current / system.m
        v_guess = v + self.dt * a_current
        r_guess = r + self.dt * v_guess
        
        # Solve the nonlinear system
        x0 = np.concatenate([r_guess, v_guess])
        solution = fsolve(residual, x0)
        
        # Extract solution
        r_new = solution[:system.nDOF]
        v_new = solution[system.nDOF:]
        
        return t_new, r_new, v_new
