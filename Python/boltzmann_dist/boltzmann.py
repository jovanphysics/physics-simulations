import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def run_simulation():
    # Prompt user for matrix dimension N
    try:
        N = int(input("Enter matrix size N (e.g., 20): "))
        if N <= 1:
            print("N must be greater than 1.")
            return
    except ValueError:
        print("Invalid input! Please enter an integer.")
        return

    num_cells = N * N
    # Step 1: Initialize NxN matrix filled with ones
    matrix = np.ones((N, N), dtype=int)
    # Flatten view for fast indexing
    flat_matrix = matrix.reshape(-1)

    # Setup matplotlib figure with two subplots
    fig, (ax_heat, ax_dist) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left Plot: Heatmap of the matrix
    im = ax_heat.imshow(matrix, cmap="YlOrRd", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax_heat)
    cbar.set_label("Energy Value")
    ax_heat.set_title("Matrix Heatmap")

    # Right Plot: Bar chart of value distribution
    unique, counts = np.unique(matrix, return_counts=True)
    ax_dist.set_xlabel("Element Value (Energy)")
    ax_dist.set_ylabel("Count (Number of Particles)")
    ax_dist.set_title("Value Distribution")

    step_counter = [0]
    steps_per_frame = 200  # Optimized step count per frame update

    def update(frame):
        # Fast vector/index-based simulation steps
        for _ in range(steps_per_frame):
            # Step 4: Find all element indices with value > 0
            nonzero_indices = np.flatnonzero(flat_matrix)
            
            if len(nonzero_indices) == 0:
                break

            # Randomly select one positive element to decrement
            idx1 = np.random.choice(nonzero_indices)

            # Step 3: Randomly pick another element (different from idx1) to increment
            idx2 = np.random.randint(0, num_cells - 1)
            if idx2 >= idx1:
                idx2 += 1

            # Execute the energy transfer directly
            flat_matrix[idx1] -= 1
            flat_matrix[idx2] += 1

            step_counter[0] += 1

        # Update Heatmap data
        im.set_data(matrix)
        im.set_clim(vmin=0, vmax=matrix.max())

        # Update Distribution Bar Chart
        vals, val_counts = np.unique(matrix, return_counts=True)
        max_val = matrix.max()

        # Calculate entropy, energy, and temperature
        vals, counts = np.unique(matrix, return_counts=True)
        probs = counts / num_cells
        entropy = -np.sum(probs * np.log(probs)) * num_cells
        mean_energy = matrix.mean()
        beta = np.log(1 + 1 / mean_energy)
        temperature = 1 / beta  
        

        # Build full frequency array from 0 to max_val
        full_counts = np.zeros(max_val + 1, dtype=int)
        full_counts[vals] = val_counts

        ax_dist.clear()
        ax_dist.bar(
            range(max_val + 1),
            full_counts,
            color="skyblue",
            edgecolor="black",
        )
        ax_dist.set_xlabel("Element Value (Energy)")
        ax_dist.set_ylabel("Count (Number of Particles)")
        ax_dist.set_title("Value Distribution")
        ax_dist.set_ylim(0, num_cells)

        # Update Main Figure Title with total step count and top padding
        fig.suptitle(
                    f"Boltzmann Simulation - Steps: {step_counter[0]:,}\n"
                    f"Entropy (S): {entropy:.1f}  |  Temperature (T): {temperature:.3f} | Mean_energy: {mean_energy:.3f} | k_B = 1",
                    fontsize=13,
                    fontweight="bold",
                    y=0.98
                )
        return im,

    # Keep a strong reference to FuncAnimation so garbage collector doesn't stop it
    ani = animation.FuncAnimation(fig, update, interval=20, cache_frame_data=False)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


if __name__ == "__main__":
    run_simulation()