import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

class AdaptiveECWFCore:
    def __init__(self, num_cognitive_dims, num_ethical_dims, num_facets, num_nodes=10):
        # ECWF Initialization
        self.num_cognitive_dims = num_cognitive_dims
        self.num_ethical_dims = num_ethical_dims
        self.num_facets = num_facets

        # Parameters
        self.k = np.random.rand(num_facets, num_cognitive_dims) * 2 - 1
        self.m = np.random.rand(num_facets, num_ethical_dims) * 2 - 1
        self.omega = np.random.rand(num_facets) * 2 * np.pi
        self.phi = np.random.rand(num_facets) * 2 * np.pi

        # Memory Web
        self.memory_web = self._initialize_memory_web(num_nodes)

    def _initialize_memory_web(self, num_nodes):
        """Initialize Memory Web with nodes and connections."""
        G = nx.erdos_renyi_graph(num_nodes, 0.3)
        for node in G.nodes():
            G.nodes[node]['cognitive_activation'] = np.random.rand()
            G.nodes[node]['ethical_activation'] = np.random.rand()
        for (u, v) in G.edges():
            G.edges[u, v]['strength'] = np.random.rand()
        return G

    def compute_ecwf(self, x, e, t):
        """Compute the ECWF value."""
        # Ensure inputs are 1D arrays
        x = np.ravel(x)
        e = np.ravel(e)

        # Initialize the wave_sum with the correct shape (2D grid)
        wave_sum = np.zeros((len(x), len(e)), dtype=np.complex128)  # Create a 2D grid

        # Compute the wave function over the facets
        for i in range(self.num_facets):
            # Ensure k[i] and m[i] can broadcast with x and e
            wave_sum += np.exp(1j * (self.k[i, 0] * x[:, None] + self.m[i, 0] * e[None, :] + self.omega[i] * t + self.phi[i]))

        return np.abs(wave_sum)  # Return the magnitude of the wave

    def update_memory_web(self, x, e, t):
        """Update the Memory Web based on ECWF dynamics."""
        Z = self.compute_ecwf(x, e, t)
        for node in self.memory_web.nodes():
            cog_act = self.memory_web.nodes[node]['cognitive_activation']
            eth_act = self.memory_web.nodes[node]['ethical_activation']
            self.memory_web.nodes[node]['cognitive_activation'] = cog_act * (1 + 0.1 * np.sin(Z.mean()))
            self.memory_web.nodes[node]['ethical_activation'] = eth_act * (1 + 0.1 * np.cos(Z.mean()))
        updated_memory_web = self.memory_web.to_directed()  # Convert to directed graph to visualize connections
        for (u, v) in updated_memory_web.edges():
            u_act = updated_memory_web.nodes[u]['cognitive_activation']
            v_act = updated_memory_web.nodes[v]['cognitive_activation']
            strength = (u_act + v_act) / 2
            updated_memory_web.edges[u, v]['strength'] = np.clip(strength, 0, 1)
        return updated_memory_web

    def visualize_memory_web(self, t, selected_node=None):
        """Visualize the Memory Web with node highlighting."""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.memory_web, seed=42)
        node_colors = [self.memory_web.nodes[n]['cognitive_activation'] for n in self.memory_web.nodes()]
        node_sizes = [500 * self.memory_web.nodes[n]['ethical_activation'] for n in self.memory_web.nodes()]

        # Create a list of edge tuples with a uniform edge color and weight
        edge_tuples = list(self.memory_web.edges)
        edge_colors = ['gray' for _ in edge_tuples]
        edge_weights = [self.memory_web.edges[u, v]['strength'] for u, v in edge_tuples]

        # If a selected_node is provided, update the corresponding edge colors and weights
        if selected_node is not None:
            selected_edges = [(u, v) for u, v in self.memory_web.edges() if selected_node in (u, v)]
            edge_colors = ['red' if (u, v) in selected_edges else 'gray' for u, v in self.memory_web.edges()]
            edge_weights = [1.0 if (u, v) in selected_edges else 0.5 for u, v in self.memory_web.edges()]

        nx.draw(self.memory_web, pos,
                node_color=node_colors,
                node_size=node_sizes,
                edge_color=edge_colors,
                width=edge_weights,
                with_labels=True,
                cmap=plt.cm.Blues)
        plt.title(f"Memory Web at Time {t:.2f} - Highlighting Node {selected_node}" if selected_node else f"Memory Web at Time {t:.2f}")
        plt.show()

    def visualize_ecwf(self, x, e, t, ax):
        """3D Visualization of the ECWF."""
        x_grid, e_grid = np.meshgrid(x, e)
        Z = self.compute_ecwf(x, e, t)
        surf = ax.plot_surface(x_grid, e_grid, Z, cmap='viridis', edgecolor='none')
        ax.set_title(f"ECWF Visualization at Time {t:.2f}")
        ax.set_xlabel('Cognitive Input')
        ax.set_ylabel('Ethical Input')
        ax.set_zlabel('Amplitude')
        return surf

def run_simulation(ecwf, num_steps=20, x_range=(-10, 10), e_range=(-10, 10), selected_node=0):
    """Run the Adaptive ECWF Simulation."""
    x = np.linspace(*x_range, 50)
    e = np.linspace(*e_range, 50)
    time_points = np.linspace(0, 50, num_steps)

    for i, t in enumerate(time_points):
        # ECWF Visualization
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121, projection='3d')
        ecwf.visualize_ecwf(x, e, t, ax)

        # Memory Web Visualization
        plt.subplot(122)
        updated_memory_web = ecwf.update_memory_web(x, e, t)
        pos = nx.spring_layout(updated_memory_web, seed=42)

        # Calculate node_colors and node_sizes here
        node_colors = [updated_memory_web.nodes[n]['cognitive_activation'] for n in updated_memory_web.nodes()]
        node_sizes = [500 * updated_memory_web.nodes[n]['ethical_activation'] for n in updated_memory_web.nodes()]

        edge_tuples = list(updated_memory_web.edges)
        edge_colors = ['gray' for _ in edge_tuples]
        edge_weights = [updated_memory_web.edges[u, v]['strength'] for u, v in edge_tuples]
        nx.draw_networkx_edges(updated_memory_web, pos, edge_color=edge_colors, width=edge_weights)
        nx.draw_networkx_nodes(updated_memory_web, pos, node_color=node_colors, node_size=node_sizes)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Main Program
if __name__ == "__main__":
    ecwf = AdaptiveECWFCore(num_cognitive_dims=2, num_ethical_dims=2, num_facets=5, num_nodes=15)
    run_simulation(ecwf, num_steps=5, x_range=(-5, 5), e_range=(-5, 5), selected_node=2)
