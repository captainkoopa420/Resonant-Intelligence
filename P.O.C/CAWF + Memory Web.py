import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit

import networkx as nx
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score


# -----------------------------
# CAWF Core Class
# -----------------------------
class CAWFCore:
    """
    A stand-alone 'Cognitive Amplitude Wave Function' core with:
      - Wave-based cognition (no explicit ethical dimension).
      - Adaptive parameters & optional memory effect.
      - A Memory Web for interpretability (node activation).
    """

    def __init__(
        self,
        num_cognitive_dims,
        num_facets,
        feedback_factor=0.05,
        adaptive_rate=0.1,
        decay_c=1.0,
        memory_weight=0.1,
        memory_length=50,
        num_nodes=10,
        random_state=None
    ):
        """
        :param num_cognitive_dims: Number of purely cognitive dimensions.
        :param num_facets: Number of wave facets.
        :param feedback_factor: Coupling factor for amplitude updates.
        :param adaptive_rate: Rate of random parameter perturbation each step.
        :param decay_c: Decay constant for the cognitive amplitude.
        :param memory_weight: Weight for wavefunction memory effect.
        :param memory_length: Number of past states to store for memory effect.
        :param num_nodes: Number of nodes in the Memory Web.
        :param random_state: Seed for reproducibility.
        """
        self.num_cognitive_dims = num_cognitive_dims
        self.num_facets = num_facets
        self.feedback_factor = feedback_factor
        self.adaptive_rate = adaptive_rate
        self.decay_c = decay_c
        self.memory_weight = memory_weight
        self.memory_length = memory_length
        self.num_nodes = num_nodes
        self.random_state = random_state

        rng = np.random.default_rng(random_state)

        # Wave parameters:
        # For CAWF, we treat wavefunction as: Ψ(x, t) = sum over i of amplitude_i * e^{i * phase_i}
        # We'll define wavevectors, freq, phases, etc.
        self.k = rng.uniform(-1, 1, size=(num_facets, num_cognitive_dims))
        self.omega = rng.uniform(0, 2*np.pi, size=num_facets)
        self.phi = rng.uniform(0, 2*np.pi, size=num_facets)

        # Memory arrays to store wavefunction states for classification vs. visualization
        self.past_states_classification = []
        self.past_states_visualization = []

        # Build a Memory Web for interpretability
        self.memory_web = self._initialize_memory_web(num_nodes, rng)

    def _initialize_memory_web(self, num_nodes, rng):
        G = nx.erdos_renyi_graph(num_nodes, 0.3, seed=self.random_state)
        for node in G.nodes():
            G.nodes[node]['cognitive_activation'] = rng.random()
        for (u, v) in G.edges():
            G.edges[u, v]['strength'] = rng.random()
        return G

    @staticmethod
    @njit
    def norm(vector):
        return np.sqrt(np.sum(vector**2, axis=-1))

    def amplitude_i(self, x, i):
        """
        Compute amplitude for facet i using exponential decay in the cognitive space, plus optional feedback.
        For pure CAWF, we can incorporate a simple tanh coupling or skip it entirely.
        """
        # L2 norm of x
        norm_x = self.norm(x)
        exp_decay_x = np.exp(-norm_x**2 / self.decay_c)

        # Simple feedback coupling (demo approach)
        # If you want a pure wave, set feedback_factor=0
        amplitude = exp_decay_x * (1 + self.feedback_factor * np.tanh(norm_x))
        return amplitude + 1e-10  # avoid zero amplitude

    def phase_i(self, x, t, i):
        """
        Compute the wave phase for facet i.
        Phase = 2π(k_i·x) - ω_i*t + φ_i
        """
        cognitive_phase = np.dot(x, self.k[i])
        phase = 2*np.pi * cognitive_phase - self.omega[i]*t + self.phi[i]
        return phase

    def compute_cawf(self, x, t, store_memory=False, memory_type='classification'):
        """
        Compute the Cognitive Amplitude Wave Function for a given x grid + time t.
        :param x: shape (M, N, num_cognitive_dims)
        :param t: scalar time
        :param store_memory: whether to store wave output in memory
        :param memory_type: 'classification' or 'visualization'
        :return: wavefunction shape (M, N) as complex array
        """
        result = np.zeros(x.shape[:-1], dtype=complex)

        for i in range(self.num_facets):
            amp = self.amplitude_i(x, i)
            phs = self.phase_i(x, t, i)

            # Adaptive parameter perturbation
            rng = np.random.default_rng(self.random_state)
            self.k[i] += self.adaptive_rate * rng.standard_normal(self.k[i].shape)
            self.omega[i] += self.adaptive_rate * rng.standard_normal()
            self.phi[i] += self.adaptive_rate * rng.standard_normal()

            result += amp * np.exp(1j * phs)

        if store_memory:
            mem_list = (
                self.past_states_classification 
                if memory_type == 'classification' 
                else self.past_states_visualization
            )
            mem_list.append(result.copy())
            if len(mem_list) > self.memory_length:
                mem_list.pop(0)

            # If shapes are consistent, apply memory effect
            first_shape = mem_list[0].shape if len(mem_list) > 0 else None
            if first_shape == result.shape:
                memory_effect = np.mean(mem_list, axis=0)
                result += self.memory_weight * memory_effect

        return result + 1e-10

    def calculate_entropy(self, psi):
        """
        Shannon entropy of wavefunction magnitude
        """
        p = np.abs(psi)**2
        p_sum = np.sum(p)
        if p_sum == 0:
            return 0.0
        p /= p_sum
        entropy = -np.nansum(p * np.log2(p + 1e-10))
        return entropy

    def update_memory_web(self, x, t):
        """
        Update memory web nodes based on average wave amplitude at the given x + time.
        """
        Z = self.compute_cawf(x, t, store_memory=False, memory_type='visualization')
        mean_amp = np.abs(Z).mean()

        for node in self.memory_web.nodes():
            cog_act = self.memory_web.nodes[node]['cognitive_activation']
            # Example: node activations grow with wave amplitude
            self.memory_web.nodes[node]['cognitive_activation'] = cog_act * (1 + 0.1*np.sin(mean_amp))

        updated_memory_web = self.memory_web.to_directed()
        for (u, v) in updated_memory_web.edges():
            u_act = updated_memory_web.nodes[u]['cognitive_activation']
            v_act = updated_memory_web.nodes[v]['cognitive_activation']
            strength = (u_act + v_act)/2
            updated_memory_web.edges[u, v]['strength'] = np.clip(strength, 0, 1)

        return updated_memory_web

    def visualize_cawf(self, x_range, t, ax):
        """
        3D surface plot for CAWF amplitude over a 2D grid of x (only if num_cognitive_dims == 2).
        """
        if self.num_cognitive_dims != 2:
            raise ValueError("visualize_cawf only works for num_cognitive_dims=2.")
        x_vals, y_vals = np.meshgrid(x_range, x_range)
        # shape: (len(x_range), len(x_range), 2)
        x_input = np.zeros((len(x_range), len(x_range), 2))
        x_input[...,0] = x_vals
        x_input[...,1] = y_vals

        Z = self.compute_cawf(x_input, t, store_memory=False, memory_type='visualization')
        surf = ax.plot_surface(x_vals, y_vals, np.abs(Z), cmap='viridis', edgecolor='none')
        ax.set_xlabel('Cognitive Dim 1')
        ax.set_ylabel('Cognitive Dim 2')
        ax.set_zlabel('|Ψ|')
        ax.set_title(f'CAWF at time = {t:.2f}')
        return surf

    def visualize_memory_web(self, t, selected_node=None):
        plt.figure(figsize=(6,5))
        pos = nx.spring_layout(self.memory_web, seed=42)
        node_colors = [self.memory_web.nodes[n]['cognitive_activation'] for n in self.memory_web.nodes()]
        node_sizes = [500 * act for act in node_colors]

        edge_tuples = list(self.memory_web.edges)
        edge_colors = ['gray'] * len(edge_tuples)
        edge_weights = [self.memory_web.edges[u,v]['strength'] for u,v in edge_tuples]

        if selected_node is not None:
            selected_edges = [(u,v) for (u,v) in edge_tuples if selected_node in (u,v)]
            edge_colors = ['red' if (u,v) in selected_edges else 'gray' for (u,v) in edge_tuples]
            edge_weights = [
                1.0 if (u,v) in selected_edges else w
                for (u,v,w) in zip(*zip(*edge_tuples), edge_weights)
            ]

        nx.draw(self.memory_web, pos,
                node_color=node_colors,
                node_size=node_sizes,
                edge_color=edge_colors,
                width=edge_weights,
                with_labels=True,
                cmap=plt.cm.Blues)
        title_str = f"Memory Web at t={t:.2f}"
        if selected_node is not None:
            title_str += f" (Node {selected_node} highlighted)"
        plt.title(title_str)
        plt.show()


# -----------------------------
# CAWFClassifier (scikit-learn)
# -----------------------------
class CAWFClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible classifier that leverages CAWFCore. 
    This allows cross-validation and direct comparisons to baseline ML models.
    """

    def __init__(
        self,
        num_cognitive_dims=2,
        num_facets=5,
        feedback_factor=0.05,
        adaptive_rate=0.1,
        decay_c=1.0,
        memory_weight=0.1,
        memory_length=50,
        num_nodes=10,
        random_state=None
    ):
        self.num_cognitive_dims = num_cognitive_dims
        self.num_facets = num_facets
        self.feedback_factor = feedback_factor
        self.adaptive_rate = adaptive_rate
        self.decay_c = decay_c
        self.memory_weight = memory_weight
        self.memory_length = memory_length
        self.num_nodes = num_nodes
        self.random_state = random_state

        self.cawf = None
        self.classes_ = None
        self.t_ = 0.0  # track time if desired

    def fit(self, X, y):
        """
        Initialize CAWFCore and store class labels.
        """
        self.cawf = CAWFCore(
            num_cognitive_dims=self.num_cognitive_dims,
            num_facets=self.num_facets,
            feedback_factor=self.feedback_factor,
            adaptive_rate=self.adaptive_rate,
            decay_c=self.decay_c,
            memory_weight=self.memory_weight,
            memory_length=self.memory_length,
            num_nodes=self.num_nodes,
            random_state=self.random_state
        )
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        predictions = []
        for row in X:
            # For classification, assume the first num_cognitive_dims are the input features
            # shape: (1,1,num_cognitive_dims)
            x_input = row[:self.num_cognitive_dims].reshape((1,1,self.num_cognitive_dims))

            Z = self.cawf.compute_cawf(
                x_input,
                t=self.t_,
                store_memory=True,     # store in classification memory
                memory_type='classification'
            )
            amplitude = np.abs(Z).mean()

            if len(self.classes_) == 2:
                # simple threshold approach
                threshold = 0.5
                if amplitude > threshold:
                    predictions.append(self.classes_[1])
                else:
                    predictions.append(self.classes_[0])
            else:
                # For multi-class, do a naive amplitude-based index
                idx = int(amplitude * 10) % len(self.classes_)
                predictions.append(self.classes_[idx])

            self.t_ += 0.05  # increment time

        return np.array(predictions)

    def predict_proba(self, X):
        """
        For 2-class scenario, map amplitude to a logistic function.
        """
        if len(self.classes_) != 2:
            # Not implemented for multi-class
            return np.zeros((len(X), len(self.classes_)))

        probs = []
        for row in X:
            x_input = row[:self.num_cognitive_dims].reshape((1,1,self.num_cognitive_dims))
            Z = self.cawf.compute_cawf(
                x_input,
                t=self.t_,
                store_memory=True,
                memory_type='classification'
            )
            amplitude = np.abs(Z).mean()

            score = amplitude - 0.5
            prob_class1 = 1.0/(1.0 + np.exp(-5 * score))
            prob_class0 = 1 - prob_class1
            probs.append([prob_class0, prob_class1])

            self.t_ += 0.05

        return np.array(probs)


# -----------------------------
# Demo / Main Execution
# -----------------------------
if __name__ == "__main__":
    from sklearn.model_selection import StratifiedKFold, train_test_split

    # Synthetic data for 2-class problem. If num_cognitive_dims=2, each sample has 2 features
    rng = np.random.default_rng(42)
    X_demo = rng.random((50, 2))  # 50 samples, each with 2 "cognitive" features
    y_demo = np.array([0 if i<25 else 1 for i in range(50)])  # half 0, half 1

    cawf_clf = CAWFClassifier(
        num_cognitive_dims=2,
        num_facets=5,
        feedback_factor=0.05,
        adaptive_rate=0.05,
        decay_c=1.0,
        memory_weight=0.1,
        memory_length=20,
        num_nodes=10,
        random_state=42
    )

    # Cross-validation
    skf = StratifiedKFold(n_splits=5)
    scores = cross_val_score(cawf_clf, X_demo, y_demo, cv=skf, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

    # Holdout example
    X_train, X_test, y_train, y_test = train_test_split(X_demo, y_demo, test_size=0.2, stratify=y_demo, random_state=42)
    cawf_clf.fit(X_train, y_train)
    y_pred = cawf_clf.predict(X_test)
    holdout_acc = accuracy_score(y_test, y_pred)
    print(f"Holdout Accuracy: {holdout_acc:.2f}")

    # Visualization if num_cognitive_dims=2
    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(1,2,1, projection='3d')

    cawf_core = cawf_clf.cawf
    # 2D wave amplitude plot
    x_range = np.linspace(-5,5,30)
    t_val = 0.0
    cawf_core.visualize_cawf(x_range, t_val, ax=ax)

    # Memory web on second subplot
    plt.subplot(1,2,2)
    # We'll do a single call for memory update
    dummy_input = np.zeros((1,1,cawf_core.num_cognitive_dims))
    cawf_core.update_memory_web(dummy_input, t_val)
    cawf_core.visualize_memory_web(t_val, selected_node=2)

    plt.tight_layout()
    plt.show()
