# Memory Web

## Definition

The Memory Web in this project represents a dynamic network of interconnected nodes, used to model the evolution of cognitive and ethical aspects of a system. It incorporates activation waves that propagate through the network, interacting to produce emergent patterns and insights. Node activations and edge strengths are dynamically updated based on these interactions, allowing the network to adapt and learn over time.

## Components

**Nodes:**
- Represent concepts, ideas, or memory elements.
- **Attributes:**
    - `cognitive_activation`: Represents the level of cognitive activity associated with the node, ranging from 0 (inactive) to 1 (fully active).
    - `ethical_activation`: Represents the ethical significance of the node, ranging from 0 (no ethical relevance) to 1 (highly ethically relevant).

**Edges:**
- Represent relationships or associations between nodes.
- **Attributes:**
    - `strength`: Represents the strength of the connection between two nodes, ranging from 0 (no connection) to 1 (strong connection).

## Initialization

- Created using an Erdős-Rényi random graph model.
- Initial node activations and edge strengths are assigned randomly.

## Dynamics

- Evolves over time based on the Ethical-Cognitive Wave Function (ECWF).
- The Ethical-Cognitive Wave Function (ECWF) governs the dynamic evolution of the Memory Web. Node activations are updated based on the interference of incoming waves from connected nodes, weighted by the connection strengths. Edge strengths are adjusted based on the co-activation patterns of connected nodes, reflecting the strengthening or weakening of associations over time.


## Visualization

- Visualized using `matplotlib.pyplot` and `networkx`.
- Node color and size reflect cognitive and ethical activations.
- Edge thickness represents connection strength.
- Selected nodes can be highlighted to visualize the propagation of activation waves, the emergence of patterns, or the influence of specific nodes on the overall network dynamics.

## Example Code

# Node colors based on activation
        node_colors = [self.graph.nodes[node]['activation'] for node in self.graph.nodes()]
        node_sizes = [1000 * self.graph.nodes[node]['activation'] for node in self.graph.nodes()]
        
        # Edge widths based on weights
        edge_weights = [self.graph.edges[edge]['weight'] * 5 for edge in self.graph.edges()]
        
        nx.draw(
            self.graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color='gray',
            width=edge_weights,
            with_labels=True,
            cmap=plt.cm.Blues
        )
        plt.title(title)
        plt.show()


# Example Usage
if __name__ == "__main__":
    memory_web = MemoryWeb(num_nodes=15)
    memory_web.visualize(title="Initial Memory Web")
    
    # Propagate wave from a source node
    source = 0
    memory_web.propagate_wave(source_node=source, wave_strength=1.0)
    memory_web.visualize(title=f"Memory Web After Wave Propagation from Node {source}")
    
    # Update the network
    memory_web.update_network()
    memory_web.visualize(title="Memory Web After Network Update")
