import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit 

# 1. Initialize the lattice grid using Numba-compatible random methods
@njit
def initialize_lattice(size):
    """
    Initializes a 2D lattice grid with random spins of 1 or -1.
    Uses np.random.rand for Numba compatibility.
    """
    return np.where(np.random.rand(size, size) < 0.5, -1, 1).astype(np.float64)

@njit
def neighbor_sum(spins, i, j):
    """
    Calculates the sum of the 4 nearest neighbors with Periodic Boundary Conditions (PBC).
    """
    N = spins.shape[0]
    return (
        spins[(i + 1) % N, j]
        + spins[(i - 1) % N, j]
        + spins[i, (j + 1) % N]
        + spins[i, (j - 1) % N]
    )

@njit
def calculate_initial_energy(spins, J, h):
    """
    Computes the total baseline energy of the initial lattice state from scratch.
    Time complexity: O(N^2) - executed once at the start.
    """
    N = spins.shape[0]
    energy = 0.0
    for i in range(N):
        for j in range(N):
            energy -= J * spins[i, j] * (
                spins[(i + 1) % N, j] + spins[i, (j + 1) % N]
            )
            energy -= h * spins[i, j]
    return energy

# 2. Function to execute 1 Monte Carlo Step (MCS)
@njit
def monte_carlo_step(spins, beta, J, h, current_energy, current_mag):
    """
    Executes a single Monte Carlo Step (N*N spin flip attempts)
    and dynamically updates system energy and magnetization.
    """
    N = spins.shape[0]
    for _ in range(N * N):
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        
        s = spins[i, j]
        dE = 2 * s * (J * neighbor_sum(spins, i, j) + h)
        
        # Metropolis acceptance criterion
        if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
            spins[i, j] = -s
            current_energy += dE
            current_mag += 2 * (-s)
            
    return current_energy, current_mag

def main():
    print("=== 2D ISING MODEL VISUAL SIMULATION (MATPLOTLIB) === \n")

    # Interactive User Configurations
    size = int(input("Enter lattice size (N): "))
    temperature = float(input("Enter temperature (T): "))
    J = float(input("Enter coupling constant (J): "))
    h = float(input("Enter external magnetic field (h): "))
    steps = int(input("Enter total simulation steps (MCS): "))

    beta = 1.0 / temperature
    spins = initialize_lattice(size)
    
    current_energy = calculate_initial_energy(spins, J, h)
    current_mag = np.sum(spins)

    # Initialize data tracking history for plotting
    energy_history = []
    mag_history = []
    steps_history = []

    # Setup Matplotlib Dashboard Layout ✨
    plt.style.use('dark_background')  # Sleek dark theme 🌙
    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f"2D Ising Model Simulation (N={size}, T={temperature}, J={J}, h={h})", fontsize=14, fontweight='bold', color='#FFD1DC')

    # Subplot 1: Lattice Spin Configuration Grid
    ax_grid = fig.add_subplot(1, 2, 1)
    im = ax_grid.imshow(spins, cmap='coolwarm', vmin=-1, vmax=1)
    ax_grid.set_title("Lattice Spin Orientation", fontsize=12)
    ax_grid.axis('off')

    # Subplot 2: Energy & Magnetization Real-Time Curves
    ax_stats = fig.add_subplot(1, 2, 2)
    line_energy, = ax_stats.plot([], [], color='#FF6B6B', label='Total Energy (E)', lw=1.8)
    
    # Secondary Y-axis for Magnetization
    ax_mag = ax_stats.twinx()
    line_mag, = ax_mag.plot([], [], color='#4D8BFF', label='Total Magnetization (M)', lw=1.8, linestyle='--')

    ax_stats.set_xlabel("Monte Carlo Steps (MCS)", fontsize=10)
    ax_stats.set_ylabel("Energy", color='#FF6B6B', fontsize=10)
    ax_mag.set_ylabel("Magnetization", color='#4D8BFF', fontsize=10)
    ax_stats.set_title("Thermodynamic Dynamics", fontsize=12)
    ax_stats.grid(True, linestyle=':', alpha=0.4)

    # Consolidate plot legends across dual Y-axes
    lines = [line_energy, line_mag]
    labels = [l.get_label() for l in lines]
    ax_stats.legend(lines, labels, loc='upper right')

    # Animation Frame Update Function
    def update(step):
        nonlocal current_energy, current_mag

        if step < steps:
            # Perform 1 MCS iteration
            current_energy, current_mag = monte_carlo_step(spins, beta, J, h, current_energy, current_mag)

            # Record iteration history
            energy_history.append(current_energy)
            mag_history.append(current_mag)
            steps_history.append(step + 1)

            # Update lattice visual representation
            im.set_array(spins)

            # Update thermodynamic trajectory plots
            line_energy.set_data(steps_history, energy_history)
            line_mag.set_data(steps_history, mag_history)

            # Dynamic axis re-scaling
            ax_stats.set_xlim(0, max(steps, step + 1))
            
            # Auto-scale Y-axis limits
            ax_stats.set_ylim(min(energy_history) - 5, max(energy_history) + 5)
            ax_mag.set_ylim(min(mag_history) - 5, max(mag_history) + 5)

        return im, line_energy, line_mag

    # Launch Real-Time Animation Loop 🚀
    ani = animation.FuncAnimation(
        fig, update, frames=steps, interval=20, blit=False, repeat=False
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()