import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, but is otherwise unused.

class AdaptiveECWFCore:
    """
    A class representing an adaptive Ethical Cognitive Wave Function (ECWF) core.

    Attributes:
        num_cognitive_dims (int): Number of cognitive dimensions.
        num_ethical_dims (int): Number of ethical dimensions.
        num_facets (int): Number of facets in the ECWF.
        k (np.ndarray): Cognitive wave vectors for each facet.
        m (np.ndarray): Ethical wave vectors for each facet.
        omega (np.ndarray): Angular frequencies for each facet.
        phi (np.ndarray): Initial phases for each facet.
        feedback_factor (float): Factor for feedback term.
        adaptive_rate (float): Rate of adaptability.
    """
    def __init__(self, num_cognitive_dims, num_ethical_dims, num_facets):
        self.num_cognitive_dims = num_cognitive_dims
        self.num_ethical_dims = num_ethical_dims
        self.num_facets = num_facets

        # Initialize parameters
        self.k = np.random.rand(num_facets, num_cognitive_dims) * 2 - 1  # Shape: (num_facets, num_cognitive_dims)
        self.m = np.random.rand(num_facets, num_ethical_dims) * 2 - 1    # Shape: (num_facets, num_ethical_dims)
        self.omega = np.random.rand(num_facets) * 2 * np.pi              # Shape: (num_facets,)
        self.phi = np.random.rand(num_facets) * 2 * np.pi                # Shape: (num_facets,)
        self.feedback_factor = 0.05
        self.adaptive_rate = 0.1

    def M_i(self, x, e, t, i):
        """
        Compute the amplitude for facet i.

        Args:
            x (np.ndarray): Cognitive input array.
            e (np.ndarray): Ethical input array.
            t (float): Time variable.
            i (int): Facet index.

        Returns:
            float: Computed amplitude.
        """
        cognitive_term = np.sum(x**2, axis=-1) / self.num_cognitive_dims
        ethical_term = np.sum(e**2, axis=-1) / self.num_ethical_dims
        feedback_term = self.feedback_factor * np.sin(np.sum(x * e, axis=-1) + t)
        adaptive_feedback = self.adaptive_rate * np.tanh(cognitive_term - ethical_term)
        amplitude = np.exp(-(cognitive_term + 2 * ethical_term) / (2 * (i + 1))) \
                    * (1 + 0.5 * np.sin(3 * t) + feedback_term + adaptive_feedback) + 1e-10
        return amplitude

    def compute_ecwf(self, x, e, t):
        """
        Compute the Ethical Cognitive Wave Function.

        Args:
            x (np.ndarray): Cognitive input array.
            e (np.ndarray): Ethical input array.
            t (float): Time variable.

        Returns:
            np.ndarray: Computed ECWF values.
        """
        result = np.zeros(x.shape[:-1], dtype=complex)
        for i in range(self.num_facets):
            amplitude = self.M_i(x, e, t, i)

            # Update parameters with correct shapes
            self.k[i] += 0.01 * np.random.randn(*self.k[i].shape) + self.adaptive_rate * np.random.randn(*self.k[i].shape)
            self.m[i] += 0.01 * np.random.randn(*self.m[i].shape) + self.adaptive_rate * np.random.randn(*self.m[i].shape)
            self.omega[i] += 0.01 * np.random.randn() + self.adaptive_rate * np.random.randn()
            self.phi[i] += 0.01 * np.random.randn() + self.adaptive_rate * np.random.randn()

            cognitive_phase = np.tensordot(x, self.k[i], axes=([-1], [-1]))  # Shape: x.shape[:-1]
            ethical_phase = np.tensordot(e, self.m[i], axes=([-1], [-1]))    # Shape: e.shape[:-1]
            phase = 2 * np.pi * (cognitive_phase + ethical_phase) - self.omega[i] * t + self.phi[i]
            result += amplitude * (np.cos(phase) + 1j * np.sin(phase))
        return result + 1e-10

    def calculate_entropy(self, psi):
        """
        Calculate the entropy of the wave function.

        Args:
            psi (np.ndarray): Wave function values.

        Returns:
            float: Calculated entropy.
        """
        p = np.abs(psi)**2
        p /= np.sum(p)
        entropy = -np.nansum(p * np.log2(p + 1e-10))
        return entropy

    def visualize_ecwf(self, x_range, e_range, t, ax):
        """
        Visualize the ECWF by plotting its magnitude.

        Args:
            x_range (np.ndarray): Range of cognitive inputs.
            e_range (np.ndarray): Range of ethical inputs.
            t (float): Time variable.
            ax (matplotlib.axes._subplots.Axes3DSubplot): Matplotlib 3D axes.

        Returns:
            Poly3DCollection: The plotted surface.
        """
        # Create grids for the first cognitive and ethical dimensions
        x_grid, e_grid = np.meshgrid(x_range, e_range)
        # Initialize cognitive and ethical inputs
        x_input = np.zeros((len(e_range), len(x_range), self.num_cognitive_dims))
        e_input = np.zeros((len(e_range), len(x_range), self.num_ethical_dims))
        # Vary the first dimension, fix others to zero
        x_input[..., 0] = x_grid
        e_input[..., 0] = e_grid
        # Compute ECWF
        Z = self.compute_ecwf(x_input, e_input, t)
        # Plot the magnitude of the ECWF
        surf = ax.plot_surface(x_grid, e_grid, np.abs(Z), cmap='viridis')
        ax.set_xlabel('Cognitive Dimension 1')
        ax.set_ylabel('Ethical Dimension 1')
        ax.set_zlabel('|Ψ|')
        return surf

def run_adaptive_ecwf_simulation(ecwf, num_steps=20, x_range=(-10, 10), e_range=(-10, 10), adaptive_threshold=0.01):
    """
    Run the adaptive ECWF simulation.

    Args:
        ecwf (AdaptiveECWFCore): The ECWF core instance.
        num_steps (int): Number of time steps.
        x_range (tuple): Range for cognitive inputs.
        e_range (tuple): Range for ethical inputs.
        adaptive_threshold (float): Threshold for adaptation decision.

    Returns:
        tuple: Entropy lists for phase one and phase two.
    """
    # Phase One: Initial Run
    fig = plt.figure(figsize=(20, 20))
    x = np.linspace(x_range[0], x_range[1], 50)
    e = np.linspace(e_range[0], e_range[1], 50)
    time_points = np.linspace(0, 50, num_steps)

    entropies_phase_one = []

    for i, t in enumerate(time_points):
        ax = fig.add_subplot(4, 5, i+1, projection='3d')
        surf = ecwf.visualize_ecwf(x, e, t, ax)

        # Prepare inputs for entropy calculation
        x_grid, e_grid = np.meshgrid(x, e)
        x_input = np.zeros((len(e), len(x), ecwf.num_cognitive_dims))
        e_input = np.zeros((len(e), len(x), ecwf.num_ethical_dims))
        x_input[..., 0] = x_grid
        e_input[..., 0] = e_grid

        Z = ecwf.compute_ecwf(x_input, e_input, t)
        entropy = ecwf.calculate_entropy(Z)
        entropies_phase_one.append(entropy)

        ax.set_title(f'Phase 1 - Time: {t:.2f}, Entropy: {entropy:.2e}')

    plt.tight_layout()
    plt.show()

    # Determine if adaptation or reset is needed
    entropy_change_phase_one = entropies_phase_one[-1] - entropies_phase_one[0]
    entropy_gradient = np.gradient(entropies_phase_one)
    adapt = np.all(np.abs(entropy_gradient) < adaptive_threshold)

    # Phase Two: Refinement or Reset
    entropies_phase_two = []
    if adapt:
        print("Adapting based on Phase One results...")
        ecwf.adaptive_rate *= 1.1  # Example adjustment: Increase adaptability rate
    else:
        print("Resetting to new base state for Phase Two...")
        ecwf.__init__(ecwf.num_cognitive_dims, ecwf.num_ethical_dims, ecwf.num_facets)  # Reinitialize

    fig = plt.figure(figsize=(20, 20))
    for i, t in enumerate(time_points):
        ax = fig.add_subplot(4, 5, i+1, projection='3d')
        surf = ecwf.visualize_ecwf(x, e, t, ax)

        # Prepare inputs for entropy calculation
        x_grid, e_grid = np.meshgrid(x, e)
        x_input = np.zeros((len(e), len(x), ecwf.num_cognitive_dims))
        e_input = np.zeros((len(e), len(x), ecwf.num_ethical_dims))
        x_input[..., 0] = x_grid
        e_input[..., 0] = e_grid

        Z = ecwf.compute_ecwf(x_input, e_input, t)
        entropy = ecwf.calculate_entropy(Z)
        entropies_phase_two.append(entropy)

        ax.set_title(f'Phase 2 - Time: {t:.2f}, Entropy: {entropy:.2e}')

    plt.tight_layout()
    plt.show()

    # Plot entropy evolution
    entropy_fig, entropy_ax = plt.subplots()
    entropy_ax.plot(time_points, entropies_phase_one, label='Phase 1 Entropy')
    entropy_ax.plot(time_points, entropies_phase_two, label='Phase 2 Entropy', linestyle='--')
    entropy_ax.set_xlabel('Time')
    entropy_ax.set_ylabel('Entropy')
    entropy_ax.set_title('Entropy Evolution Across Phases')
    entropy_ax.legend()

    plt.show()

    return entropies_phase_one, entropies_phase_two

# Usage
ecwf = AdaptiveECWFCore(num_cognitive_dims=2, num_ethical_dims=2, num_facets=5)
entropies_phase_one, entropies_phase_two = run_adaptive_ecwf_simulation(
    ecwf,
    num_steps=20,
    x_range=(-10, 10),
    e_range=(-10, 10)
)

# Analysis
print(f"Phase One - Initial Entropy: {entropies_phase_one[0]:.2e}")
print(f"Phase One - Final Entropy: {entropies_phase_one[-1]:.2e}")
print(f"Phase Two - Initial Entropy: {entropies_phase_two[0]:.2e}")
print(f"Phase Two - Final Entropy: {entropies_phase_two[-1]:.2e}")
