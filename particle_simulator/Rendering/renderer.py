import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import taichi as ti

class BaseRenderer:
    """
    Base class for renderers.
    """
    def __init__(self):
        """Initialize the base renderer."""
        pass
    
    def render(self, system, t, r, v):
        """
        Render the system state.
        
        Args:
            system (System): The system being simulated
            t (np.ndarray): Time array
            r (np.ndarray): Position array
            v (np.ndarray): Velocity array
        """
        raise NotImplementedError("Renderer is an abstract class")

class MatplotlibRenderer(BaseRenderer):
    """
    Renderer using Matplotlib for visualization.
    """
    def __init__(self, figsize=(10, 8), particle_size=50, trail_length=10):
        """
        Initialize a Matplotlib renderer.
        
        Args:
            figsize (tuple): Figure size
            particle_size (float): Size of particles in visualization
            trail_length (int): Length of particle trails
        """
        super().__init__()
        self.figsize = figsize
        self.particle_size = particle_size
        self.trail_length = trail_length
        self.fig = None
        self.ax = None
    
    def setup_3d_plot(self):
        """Set up 3D plot for visualization."""
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
    def render_frame(self, system, r_frame, v_frame=None, interactions=True):
        """
        Render a single frame.
        
        Args:
            system (System): The system being simulated
            r_frame (np.ndarray): Position array for this frame
            v_frame (np.ndarray, optional): Velocity array for this frame
            interactions (bool): Whether to render interactions
            
        Returns:
            list: List of artists for animation
        """
        artists = []
        
        # Clear the plot
        self.ax.clear()
        
        # Extract particle positions
        positions = []
        for i, particle in enumerate(system.particles):
            pos = r_frame[particle.DOF]
            positions.append(pos)
        
        positions = np.array(positions)
        
        # Plot particles
        scatter = self.ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            s=self.particle_size, c=range(len(positions)), 
            cmap='viridis', alpha=0.8
        )
        artists.append(scatter)
        
        # Draw connections/springs
        if interactions:
            for interaction in system.interactions:
                p1 = interaction.particle1
                p2 = interaction.particle2
                pos1 = r_frame[p1.DOF]
                pos2 = r_frame[p2.DOF]
                line, = self.ax.plot(
                    [pos1[0], pos2[0]], 
                    [pos1[1], pos2[1]], 
                    [pos1[2], pos2[2]], 
                    'k-', alpha=0.5
                )
                artists.append(line)
        
        # Set axis limits
        all_positions = np.vstack(positions)
        max_range = np.max([
            np.ptp(all_positions[:, 0]),
            np.ptp(all_positions[:, 1]),
            np.ptp(all_positions[:, 2])
        ])
        
        mid_x = np.mean(all_positions[:, 0])
        mid_y = np.mean(all_positions[:, 1])
        mid_z = np.mean(all_positions[:, 2])
        
        self.ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        self.ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        self.ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        return artists
        
    def render(self, system, t, r, v, show_animation=True, save_path=None, fps=30):
        """
        Render the system simulation.
        
        Args:
            system (System): The system being simulated
            t (np.ndarray): Time array
            r (np.ndarray): Position array
            v (np.ndarray): Velocity array
            show_animation (bool): Whether to display the animation
            save_path (str, optional): Path to save animation
            fps (int): Frames per second for animation
        """
        # Set up the plot
        self.setup_3d_plot()
        
        # Calculate frame interval and reduce frames if needed
        interval = 1000 / fps
        max_frames = 300  # Limit to avoid memory issues
        
        if len(t) > max_frames:
            step = len(t) // max_frames
            t_subset = t[::step]
            r_subset = r[::step]
            v_subset = v[::step]
        else:
            t_subset = t
            r_subset = r
            v_subset = v
        
        # Create animation function
        def update(frame):
            return self.render_frame(system, r_subset[frame], v_subset[frame])
        
        # Create the animation
        ani = FuncAnimation(
            self.fig, update, frames=len(t_subset),
            interval=interval, blit=True
        )
        
        # Save animation if path is provided
        if save_path:
            ani.save(save_path, fps=fps, extra_args=['-vcodec', 'libx264'])
        
        # Show animation
        if show_animation:
            plt.tight_layout()
            plt.show()
        
        return ani

class TaichiRenderer:
    """
    Real-time renderer using Taichi GUI.
    """
    def __init__(self, window_size=512, particle_radius=0.01, background_color=0x112F41):
        """
        Initialize a Taichi renderer.
        
        Args:
            window_size (int): Size of window in pixels
            particle_radius (float): Radius of particles in normalized coordinates
            background_color (int): Background color in hex
        """
        self.window_size = window_size
        self.particle_radius = particle_radius
        self.background_color = background_color
        
    def render_taichi_system(self, taichi_system, step_callback=None):
        """
        Render a Taichi system in real-time.
        
        Args:
            taichi_system (TaichiSystem): The Taichi system to render
            step_callback (callable, optional): Callback function for each step
        """
        # Initialize window
        gui = ti.GUI('Particle Simulation', (self.window_size, self.window_size),
                    background_color=self.background_color)
        
        # Determine position bounds for visualization
        @ti.kernel
        def get_bounds() -> ti.types.vector(6, ti.f32):
            # [min_x, min_y, min_z, max_x, max_y, max_z]
            result = ti.Vector([float('inf'), float('inf'), float('inf'),
                              -float('inf'), -float('inf'), -float('inf')])
            
            for i in range(taichi_system.n_particles):
                pos = taichi_system.pos[0, i]
                
                # Update mins
                result[0] = ti.min(result[0], pos[0])
                result[1] = ti.min(result[1], pos[1])
                result[2] = ti.min(result[2], pos[2])
                
                # Update maxs
                result[3] = ti.max(result[3], pos[0])
                result[4] = ti.max(result[4], pos[1])
                result[5] = ti.max(result[5], pos[2])
                
            return result
        
        bounds = get_bounds()
        
        # Add some margin
        center_x = (bounds[0] + bounds[3]) / 2
        center_y = (bounds[1] + bounds[4]) / 2
        center_z = (bounds[2] + bounds[5]) / 2
        
        max_range = max(bounds[3]-bounds[0], bounds[4]-bounds[1], bounds[5]-bounds[2])
        margin = max_range * 0.1
        
        # Function to normalize coordinates for display
        def normalize_pos(pos):
            # Map from 3D world to 2D screen
            # We'll project onto the XY plane (ignore Z for simplicity)
            x = (pos[0] - center_x + max_range/2 + margin) / (max_range + 2*margin)
            y = (pos[1] - center_y + max_range/2 + margin) / (max_range + 2*margin)
            return x, y
        
        # Render each frame
        for step in range(taichi_system.n_steps):
            # Get positions for this step
            positions = []
            for i in range(taichi_system.n_particles):
                pos = taichi_system.pos[step, i].to_numpy()
                positions.append(normalize_pos(pos))
            
            # Get connections
            connections = []
            for i in range(taichi_system.n_interactions):
                p1 = taichi_system.interaction_connections[i, 0]
                p2 = taichi_system.interaction_connections[i, 1]
                connections.append((p1, p2))
            
            # Clear canvas
            gui.clear()
            
            # Draw connections (springs)
            for p1_idx, p2_idx in connections:
                p1_pos = positions[p1_idx]
                p2_pos = positions[p2_idx]
                gui.line(p1_pos, p2_pos, radius=1, color=0x4682B4)
            
            # Draw particles
            gui.circles(positions, radius=self.particle_radius * self.window_size, color=0xFFB6C1)
            
            # Show frame
            gui.show()
            
            # Call step callback if provided
            if step_callback:
                step_callback(step)
