import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit

import networkx as nx
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

# -------------------------
# UnifiedECWFCore Class
# -------------------------

class UnifiedECWFCore:
    """
    A unified class combining:
      - Adaptive wave-based ECWF logic (amplitude, phase, memory effect)
      - A Memory Web for interpretability (graph structure).
    """

    def __init__(
        self,
        num_cognitive_dims,
        num_ethical_dims,
        num_facets,
        feedback_factor=0.05,
        adaptive_rate=0.1,
        decay_c=1.0,
        decay_e=1.0,
        memory_weight=0.1,
        memory_length=50,
        num_nodes=10,
        random_state=None
    ):
        """
        :param num_cognitive_dims: Number of cognitive dimensions.
        :param num_ethical_dims: Number of ethical dimensions.
        :param num_facets: Number of wave facets.
        :param feedback_factor: Coupling factor for amplitude updates.
        :param adaptive_rate: Rate of random perturbation applied to wave params each step.
        :param decay_c: Decay constant for cognitive amplitude.
        :param decay_e: Decay constant for ethical amplitude.
        :param memory_weight: Weighting factor for wavefunction memory effect.
        :param memory_length: Number of past states to store in memory.
        :param num_nodes: Number of nodes in the Memory Web (for interpretability).
        :param random_state: Seed for reproducibility.
        """
        self.num_cognitive_dims = num_cognitive_dims
        self.num_ethical_dims = num_ethical_dims
        self.num_facets = num_facets
        self.feedback_factor = feedback_factor
        self.adaptive_rate = adaptive_rate
        self.decay_c = decay_c
        self.decay_e = decay_e
        self.memory_weight = memory_weight
        self.memory_length = memory_length
        self.num_nodes = num_nodes
        self.random_state = random_state

        rng = np.random.default_rng(random_state)

        # Wave parameters
        self.k = rng.uniform(-1, 1, size=(num_facets, num_cognitive_dims))
        self.m = rng.uniform(-1, 1, size=(num_facets, num_ethical_dims))
        self.omega = rng.uniform(0, 2*np.pi, size=num_facets)
        self.phi = rng.uniform(0, 2*np.pi, size=num_facets)

        # Two separate memory lists:
        #   1) For classification calls (likely single-sample shapes)
        #   2) For visualization or large-grid calls (if we ever want memory there)
        self.past_states_classification = []
        self.past_states_visualization = []

        # Memory Web (NetworkX)
        self.memory_web = self._initialize_memory_web(num_nodes, rng)

    def _initialize_memory_web(self, num_nodes, rng):
        """Create a random Erdos-Renyi graph for interpretability."""
        G = nx.erdos_renyi_graph(num_nodes, 0.3, seed=self.random_state)
        for node in G.nodes():
            G.nodes[node]['cognitive_activation'] = rng.random()
            G.nodes[node]['ethical_activation'] = rng.random()
        for (u, v) in G.edges():
            G.edges[u, v]['strength'] = rng.random()
        return G

    @staticmethod
    @njit
    def norm(vector):
        return np.sqrt(np.sum(vector ** 2, axis=-1))

    def A_i(self, x, e, t, i):
        """
        Compute the amplitude for facet i using exponential decays and a tanh coupling.
        """
        norm_x = self.norm(x)
        norm_e = self.norm(e)

        exp_decay_x = np.exp(-norm_x**2 / self.decay_c)
        exp_decay_e = np.exp(-norm_e**2 / self.decay_e)

        # Broadcast shapes
        max_dims = max(self.num_cognitive_dims, self.num_ethical_dims)
        x_padded = x
        e_padded = e
        if x.shape[-1] < max_dims:
            pad_width = ((0,0),(0,0),(0, max_dims - x.shape[-1]))
            x_padded = np.pad(x, pad_width, mode='constant')
        if e.shape[-1] < max_dims:
            pad_width = ((0,0),(0,0),(0, max_dims - e.shape[-1]))
            e_padded = np.pad(e, pad_width, mode='constant')

        interaction = np.sum(x_padded * e_padded, axis=-1)
        tanh_term = np.tanh(interaction)

        amplitude = exp_decay_x * exp_decay_e * (1 + self.feedback_factor * tanh_term)
        return amplitude + 1e-10  # prevent zero amplitude

    def Phi_i(self, x, e, t, i):
        """
        Compute the phase for facet i.
        """
        cognitive_phase = np.dot(x, self.k[i])
        ethical_phase  = np.dot(e, self.m[i])
        phase = 2*np.pi*(cognitive_phase + ethical_phase) - self.omega[i]*t + self.phi[i]
        return phase

    def compute_ecwf(self, x, e, t, store_memory=False, memory_type='classification'):
        """
        Compute the Ethical Cognitive Wave Function.
        Allows optional memory storage to separate shapes for classification vs. visualization.

        :param x: shape (M, N, num_cognitive_dims)
        :param e: shape (M, N, num_ethical_dims)
        :param t: time
        :param store_memory: boolean, whether to store wave output for memory effect
        :param memory_type: 'classification' or 'visualization' to decide which memory list to use
        :return: complex wave array shape (M, N)
        """
        result = np.zeros(x.shape[:-1], dtype=complex)

        for i in range(self.num_facets):
            amplitude = self.A_i(x, e, t, i)
            phase = self.Phi_i(x, e, t, i)

            # Perturb parameters for adaptiveness
            rng = np.random.default_rng(self.random_state)
            self.k[i] += self.adaptive_rate * rng.standard_normal(self.k[i].shape)
            self.m[i] += self.adaptive_rate * rng.standard_normal(self.m[i].shape)
            self.omega[i] += self.adaptive_rate * rng.standard_normal()
            self.phi[i] += self.adaptive_rate * rng.standard_normal()

            result += amplitude * np.exp(1j * phase)

        if store_memory:
            # Decide which memory list we use
            if memory_type == 'classification':
                mem_list = self.past_states_classification
            else:
                mem_list = self.past_states_visualization

            mem_list.append(result.copy())
            if len(mem_list) > self.memory_length:
                mem_list.pop(0)

            # Only compute memory effect if shapes match the first entry
            # to avoid shape mismatch
            first_shape = mem_list[0].shape if len(mem_list) > 0 else None
            if first_shape == result.shape:
                # All stored states have consistent shape
                memory_effect = np.mean(mem_list, axis=0)
                result += self.memory_weight * memory_effect
            # else: shapes differ, skip memory effect

        return result + 1e-10

    def calculate_entropy(self, psi):
        """
        Compute the entropy of the wave function (Shannon).
        """
        p = np.abs(psi)**2
        p_sum = np.sum(p)
        if p_sum == 0:
            return 0
        p /= p_sum
        entropy = -np.nansum(p * np.log2(p+1e-10))
        return entropy

    def update_memory_web(self, x, e, t):
        """
        Update the memory web's node attributes based on wavefunction amplitude (no memory stored).
        We do a quick call with store_memory=False to avoid messing classification memory shapes.
        """
        Z = self.compute_ecwf(x, e, t, store_memory=False, memory_type='visualization')
        mean_amp = np.abs(Z).mean()

        for node in self.memory_web.nodes():
            cog_act = self.memory_web.nodes[node]['cognitive_activation']
            eth_act = self.memory_web.nodes[node]['ethical_activation']
            self.memory_web.nodes[node]['cognitive_activation'] = cog_act * (1 + 0.1*np.sin(mean_amp))
            self.memory_web.nodes[node]['ethical_activation'] = eth_act * (1 + 0.1*np.cos(mean_amp))

        updated_memory_web = self.memory_web.to_directed()
        for (u, v) in updated_memory_web.edges():
            u_act = updated_memory_web.nodes[u]['cognitive_activation']
            v_act = updated_memory_web.nodes[v]['cognitive_activation']
            strength = (u_act + v_act)/2
            updated_memory_web.edges[u, v]['strength'] = np.clip(strength, 0, 1)

        return updated_memory_web

    def visualize_ecwf(self, x_range, e_range, t, ax):
        """
        3D surface plot of the wavefunction amplitude with store_memory=False (no shape mixing).
        """
        x_grid, e_grid = np.meshgrid(x_range, e_range)
        x_input = np.zeros((len(e_range), len(x_range), self.num_cognitive_dims))
        e_input = np.zeros((len(e_range), len(x_range), self.num_ethical_dims))

        x_input[..., 0] = x_grid
        e_input[..., 0] = e_grid

        # No memory storing to avoid mixing shapes
        Z = self.compute_ecwf(x_input, e_input, t, store_memory=False, memory_type='visualization')
        surf = ax.plot_surface(x_grid, e_grid, np.abs(Z), cmap='viridis', edgecolor='none')
        ax.set_xlabel('Cognitive Dim')
        ax.set_ylabel('Ethical Dim')
        ax.set_zlabel('|Ψ|')
        ax.set_title(f'ECWF at t={t:.2f}')
        return surf

    def visualize_memory_web(self, t, selected_node=None):
        """
        Visualize the Memory Web in a 2D network layout.
        """
        plt.figure(figsize=(6, 5))
        pos = nx.spring_layout(self.memory_web, seed=42)
        node_colors = [self.memory_web.nodes[n]['cognitive_activation'] for n in self.memory_web.nodes()]
        node_sizes = [500 * self.memory_web.nodes[n]['ethical_activation'] for n in self.memory_web.nodes()]

        edge_tuples = list(self.memory_web.edges)
        edge_colors = ['gray'] * len(edge_tuples)
        edge_weights = [self.memory_web.edges[u, v]['strength'] for u,v in edge_tuples]

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


# -------------------------
# scikit-learn Wrapper
# -------------------------

class ECWFClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible classifier that uses UnifiedECWFCore internally.
    This allows cross-validation, comparisons to baseline ML, etc.
    """

    def __init__(
        self,
        num_cognitive_dims=3,
        num_ethical_dims=3,
        num_facets=5,
        feedback_factor=0.05,
        adaptive_rate=0.1,
        decay_c=1.0,
        decay_e=1.0,
        memory_weight=0.1,
        memory_length=50,
        num_nodes=10,
        random_state=None
    ):
        self.num_cognitive_dims = num_cognitive_dims
        self.num_ethical_dims = num_ethical_dims
        self.num_facets = num_facets
        self.feedback_factor = feedback_factor
        self.adaptive_rate = adaptive_rate
        self.decay_c = decay_c
        self.decay_e = decay_e
        self.memory_weight = memory_weight
        self.memory_length = memory_length
        self.num_nodes = num_nodes
        self.random_state = random_state

        self.ecwf = None
        self.classes_ = None
        self.t_ = 0.0  # keep a time variable for wavefunction if needed

    def fit(self, X, y):
        """
        Fit the wavefunction approach to the data.
        """
        self.ecwf = UnifiedECWFCore(
            num_cognitive_dims=self.num_cognitive_dims,
            num_ethical_dims=self.num_ethical_dims,
            num_facets=self.num_facets,
            feedback_factor=self.feedback_factor,
            adaptive_rate=self.adaptive_rate,
            decay_c=self.decay_c,
            decay_e=self.decay_e,
            memory_weight=self.memory_weight,
            memory_length=self.memory_length,
            num_nodes=self.num_nodes,
            random_state=self.random_state
        )
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """
        Convert the wavefunction amplitude & phase into discrete labels.
        We'll do a simple amplitude threshold for 2-class scenario.
        """
        predictions = []
        for x_row in X:
            x_cog = x_row[:self.num_cognitive_dims].reshape((1,1,self.num_cognitive_dims))
            x_eth = x_row[self.num_cognitive_dims:self.num_cognitive_dims+self.num_ethical_dims].reshape((1,1,self.num_ethical_dims))

            # For classification calls, store_memory=True, memory_type='classification'
            Z = self.ecwf.compute_ecwf(x_cog, x_eth, self.t_, store_memory=True, memory_type='classification')
            amplitude = np.abs(Z).mean()

            if len(self.classes_) == 2:
                threshold = 0.5
                if amplitude > threshold:
                    predictions.append(self.classes_[1])
                else:
                    predictions.append(self.classes_[0])
            else:
                # If multi-class, use amplitude to pick an index (rough approach)
                idx = int((amplitude*10) % len(self.classes_))
                predictions.append(self.classes_[idx])

            # Update time for demonstration
            self.t_ += 0.05

        return np.array(predictions)

    def predict_proba(self, X):
        """
        Map amplitude -> logistic function for 2-class probability.
        """
        if len(self.classes_) != 2:
            # Not implemented for multi-class
            probs = np.zeros((len(X), len(self.classes_)))
            return probs

        probabilities = []
        for x_row in X:
            x_cog = x_row[:self.num_cognitive_dims].reshape((1,1,self.num_cognitive_dims))
            x_eth = x_row[self.num_cognitive_dims:self.num_cognitive_dims+self.num_ethical_dims].reshape((1,1,self.num_ethical_dims))

            Z = self.ecwf.compute_ecwf(x_cog, x_eth, self.t_, store_memory=True, memory_type='classification')
            amplitude = np.abs(Z).mean()

            score = amplitude - 0.5
            prob_class1 = 1.0 / (1.0 + np.exp(-5*score))
            prob_class0 = 1 - prob_class1
            probabilities.append([prob_class0, prob_class1])

            self.t_ += 0.05

        return np.array(probabilities)

# -------------------------
# DEMO
# -------------------------
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    # Synthetic data: 2-class scenario, 3 cognitive + 3 ethical = 6 features total
    rng = np.random.default_rng(42)
    X_demo = rng.random((50, 6))  # 50 samples
    y_demo = np.array([0 if i<25 else 1 for i in range(50)])  # simple binary labels

    ecwf_clf = ECWFClassifier(
        num_cognitive_dims=3,
        num_ethical_dims=3,
        num_facets=5,
        feedback_factor=0.05,
        adaptive_rate=0.05,
        memory_weight=0.1,
        memory_length=20,
        num_nodes=15,
        random_state=42
    )

    # Cross-validation
    skf = StratifiedKFold(n_splits=5)
    scores = cross_val_score(ecwf_clf, X_demo, y_demo, cv=skf, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {scores.mean():.2f} ± {scores.std():.2f}")

    # Single pass holdout for illustration
    X_train, X_test, y_train, y_test = train_test_split(X_demo, y_demo, test_size=0.2, stratify=y_demo, random_state=42)
    ecwf_clf.fit(X_train, y_train)
    y_pred = ecwf_clf.predict(X_test)
    holdout_acc = accuracy_score(y_test, y_pred)
    print(f"Holdout Accuracy: {holdout_acc:.2f}")

    # Visualization demonstration
    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(1,2,1, projection='3d')

    # We'll visualize wavefunction on a 2D grid with store_memory=False
    ecwf_core = ecwf_clf.ecwf
    x_range = np.linspace(-5,5,30)
    e_range = np.linspace(-5,5,30)
    t_val = 0.0
    ecwf_core.visualize_ecwf(x_range, e_range, t_val, ax)

    # Memory web visualization (also store_memory=False internally)
    plt.subplot(1,2,2)
    ecwf_core.update_memory_web(
        np.zeros((1,1,ecwf_core.num_cognitive_dims)),
        np.zeros((1,1,ecwf_core.num_ethical_dims)),
        t_val
    )
    ecwf_core.visualize_memory_web(t_val, selected_node=2)

    plt.tight_layout()
    plt.show()
