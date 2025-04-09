import numpy as np

class Integrator:
    """
    Base class for numerical integrators.
    
    An integrator advances the state of a system through time.
    """
    def __init__(self, dt):
        """
        Initialize an integrator.
        
        Args:
            dt (float): Time step size
        """
        self.dt = dt
    
    def step(self, t, r, v, system):
        """
        Advance the system by one time step.
        
        Args:
            t (float): Current time
            r (np.ndarray): Current position vector
            v (np.ndarray): Current velocity vector
            system (System): The system to simulate
            
        Returns:
            tuple: New time, position vector, and velocity vector
        """
        raise NotImplementedError("Integrator is an abstract class")
    
    def integrate(self, system, tf):
        """
        Integrate the system from t0 to tf.
        
        Args:
            system (System): The system to simulate
            tf (float): Final time
            
        Returns:
            tuple: Time array, position array, and velocity array
        """
        t0 = system.t0
        n_steps = int((tf - t0) / self.dt) + 1
        
        # Create arrays to store results
        t = np.linspace(t0, tf, n_steps)
        r = np.zeros((n_steps, system.nDOF))
        v = np.zeros((n_steps, system.nDOF))
        
        # Set initial conditions
        r[0] = system.r0
        v[0] = system.v0
        
        # Apply constraints to initial condition
        for constraint in getattr(system, 'constraints', []):
            r[0], v[0] = constraint.apply(t[0], r[0], v[0])
        
        # Integrate step by step
        for i in range(n_steps - 1):
            t[i+1], r[i+1], v[i+1] = self.step(t[i], r[i], v[i], system)
            
            # Apply constraints after each step
            for constraint in getattr(system, 'constraints', []):
                r[i+1], v[i+1] = constraint.apply(t[i+1], r[i+1], v[i+1])
        
        return t, r, v
