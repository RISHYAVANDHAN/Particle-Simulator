import numpy as np
import taichi as ti
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict

def render_with_matplotlib(result):
    """Render the simulation using Matplotlib animation"""
    t, r, *_ = result  # Unpack the tuple (assuming result is a tuple like (t, r))
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    particles, = ax.plot([], [], 'bo', markersize=8)

    def init():
        particles.set_data([], [])
        return particles,

    def update(frame):
        if r.ndim == 3:
            x = r[frame, :, 0]
            y = r[frame, :, 1]
        else:
            x = r[frame, 0::3]
            y = r[frame, 1::3]
        particles.set_data(x, y)
        return particles,

    ani = FuncAnimation(fig, update, frames=len(t),
                        init_func=init, blit=True, interval=20)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Particle Simulation (Matplotlib)')
    plt.grid(True)
    plt.show()
    return ani

def render_with_taichi(result):
    """Render the simulation using Taichi GUI"""
    t, r, *_ = result  # Unpack the tuple (assuming result is a tuple like (t, r))
    num_particles = r.shape[1] if r.ndim == 3 else r.shape[1] // 3

    gui = ti.GUI("Particle Simulation (Taichi GUI)", res=(600, 600), background_color=0x112F41)

    frame = 0
    while gui.running and frame < len(t):
        if r.ndim == 3:
            positions = r[frame, :, :2]  # Use only x, y
        else:
            positions = np.zeros((num_particles, 2))
            for i in range(num_particles):
                positions[i, 0] = r[frame, 3 * i]
                positions[i, 1] = r[frame, 3 * i + 1]

        # Normalize for GUI (assumes bounding box of [-2, 2] x [-2, 2])
        norm_pos = 0.5 * (positions / 2.0 + 1.0)
        gui.circles(norm_pos, radius=5, color=0x068587)
        gui.show()
        frame += 1