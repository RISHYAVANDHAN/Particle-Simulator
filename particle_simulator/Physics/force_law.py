import numpy as np

class ForceLaw:
    """
    Base class for all force laws.
    
    A force law defines how particles interact with each other.
    Force laws should implement the la() method, which computes
    the force magnitude based on length, velocity, and time.
    """
    def la(self, t, l, l_dot):
        """
        Compute force based on current time, length, and length derivative.
        
        Args:
            t (float): Current time
            l (float): Current length
            l_dot (float): Current length derivative (velocity)
        
        Returns:
            float: Force magnitude
        """
        raise NotImplementedError("ForceLaw is an abstract class")
    
    def get_type_id(self):
        """
        Get the type ID for Taichi acceleration.
        
        Returns:
            int: Force law type ID
        """
        return -1  # Base class has no type
    
    def get_parameters(self):
        """
        Get force law parameters for Taichi acceleration.
        
        Returns:
            list: List of parameters
        """
        return [self.get_type_id(), 0, 0, 0, 0]

class SpringForceLaw(ForceLaw):
    """
    Hooke's law spring force.
    
    F = -k(l - l0) - c*l_dot
    
    Attributes:
        k (float): Spring stiffness
        l0 (float): Rest length
        damping (float): Damping coefficient
    """
    def __init__(self, stiffness, rest_length, damping=0.0):
        """
        Initialize a spring force law.
        
        Args:
            stiffness (float): Spring stiffness constant
            rest_length (float): Rest length of spring
            damping (float, optional): Damping coefficient. Defaults to 0.0.
        """
        self.k = stiffness
        self.l0 = rest_length
        self.damping = damping
    
    def la(self, t, l, l_dot):
        """
        Compute spring force.
        
        Args:
            t (float): Current time (not used for basic spring)
            l (float): Current length
            l_dot (float): Current length derivative
        
        Returns:
            float: Force magnitude
        """
        return -self.k * (l - self.l0) - self.damping * l_dot
    
    def get_type_id(self):
        """
        Get the type ID for Taichi acceleration.
        
        Returns:
            int: Force law type ID (0 for spring)
        """
        return 0
    
    def get_parameters(self):
        """
        Get spring parameters for Taichi acceleration.
        
        Returns:
            list: [type_id, stiffness, rest_length, damping, 0]
        """
        return [self.get_type_id(), self.k, self.l0, self.damping, 0.0]

class NonlinearSpringForceLaw(ForceLaw):
    """
    Nonlinear spring force law.
    
    F = -k(l - l0)^power - c*l_dot
    
    Attributes:
        k (float): Spring stiffness
        l0 (float): Rest length
        power (float): Exponent for nonlinearity
        damping (float): Damping coefficient
    """
    def __init__(self, stiffness, rest_length, power=2.0, damping=0.0):
        """
        Initialize a nonlinear spring force law.
        
        Args:
            stiffness (float): Spring stiffness constant
            rest_length (float): Rest length of spring
            power (float, optional): Exponent for nonlinearity. Defaults to 2.0.
            damping (float, optional): Damping coefficient. Defaults to 0.0.
        """
        self.k = stiffness
        self.l0 = rest_length
        self.power = power
        self.damping = damping
    
    def la(self, t, l, l_dot):
        """
        Compute nonlinear spring force.
        
        Args:
            t (float): Current time (not used for basic spring)
            l (float): Current length
            l_dot (float): Current length derivative
        
        Returns:
            float: Force magnitude
        """
        # Calculate displacement
        displacement = l - self.l0
        
        # Compute force based on power law
        force = -self.k * np.sign(displacement) * np.abs(displacement)**self.power
        
        # Add damping
        force -= self.damping * l_dot
        
        return force
    
    def get_type_id(self):
        """
        Get the type ID for Taichi acceleration.
        
        Returns:
            int: Force law type ID (1 for nonlinear spring)
        """
        return 1
    
    def get_parameters(self):
        """
        Get nonlinear spring parameters for Taichi acceleration.
        
        Returns:
            list: [type_id, stiffness, rest_length, damping, power]
        """
        return [self.get_type_id(), self.k, self.l0, self.damping, self.power]

class TimeDependentSpringForceLaw(ForceLaw):
    """
    Time-dependent spring force law.
    
    F = -k(t)(l - l0) - c*l_dot
    
    where k(t) can vary with time according to a given function.
    
    Attributes:
        base_stiffness (float): Base spring stiffness
        rest_length (float): Rest length
        stiffness_func (callable): Function that modifies stiffness based on time
        damping (float): Damping coefficient
    """
    def __init__(self, base_stiffness, rest_length, stiffness_func=None, damping=0.0):
        """
        Initialize a time-dependent spring force law.
        
        Args:
            base_stiffness (float): Base spring stiffness constant
            rest_length (float): Rest length of spring
            stiffness_func (callable, optional): Function to modify stiffness over time.
                                                Default is None (constant stiffness).
            damping (float, optional): Damping coefficient. Defaults to 0.0.
        """
        self.base_k = base_stiffness
        self.l0 = rest_length
        self.stiffness_func = stiffness_func or (lambda t: 1.0)  # Default to constant
        self.damping = damping
    
    def la(self, t, l, l_dot):
        """
        Compute time-dependent spring force.
        
        Args:
            t (float): Current time
            l (float): Current length
            l_dot (float): Current length derivative
        
        Returns:
            float: Force magnitude
        """
        # Get current stiffness based on time
        current_k = self.base_k * self.stiffness_func(t)
        
        # Calculate force with current stiffness
        return -current_k * (l - self.l0) - self.damping * l_dot
