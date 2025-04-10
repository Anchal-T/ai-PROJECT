import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO

def main():
    st.title("Graph Neighborhood Explorer")
    st.write("""
    This application allows you to explore neighborhoods in a graph.
    Upload a graph or generate a random one, then select a node to see its neighborhood.
    """)
    
    # Sidebar for graph creation options
    st.sidebar.header("Graph Creation")
    graph_source = st.sidebar.radio(
        "Select graph source:",
        ["Generate Random Graph", "Upload Edge List", "Upload Adjacency Matrix"]
    )
    
    G = None  # Initialize graph
    
    # Generate or load graph based on user selection
    if graph_source == "Generate Random Graph":
        G = create_random_graph()
    elif graph_source == "Upload Edge List":
        G = load_from_edge_list()
    elif graph_source == "Upload Adjacency Matrix":
        G = load_from_adjacency_matrix()
    
    # If we have a valid graph, proceed with visualization
    if G is not None and len(G.nodes) > 0:
        visualize_graph(G)
        explore_neighborhood(G)

def create_random_graph():
    """Generate a random graph based on user parameters."""
    st.sidebar.subheader("Random Graph Parameters")
    
    graph_type = st.sidebar.selectbox(
        "Graph Type",
        ["Erdős-Rényi", "Barabási-Albert", "Watts-Strogatz", "Grid", "Cycle"]
    )
    
    n_nodes = st.sidebar.slider("Number of nodes", 5, 50, 20)
    
    G = None
    
    if graph_type == "Erdős-Rényi":
        # Probability of edge creation
        p = st.sidebar.slider("Edge probability", 0.0, 1.0, 0.2, 0.05)
        G = nx.erdos_renyi_graph(n_nodes, p)
        
    elif graph_type == "Barabási-Albert":
        # Number of edges to attach from a new node to existing nodes
        m = st.sidebar.slider("Edges per new node", 1, min(5, n_nodes-1), 2)
        G = nx.barabasi_albert_graph(n_nodes, m)
        
    elif graph_type == "Watts-Strogatz":
        # K nearest neighbors to connect to
        k = st.sidebar.slider("K neighbors", 2, min(10, n_nodes-1), 4)
        # Rewiring probability
        p = st.sidebar.slider("Rewiring probability", 0.0, 1.0, 0.3, 0.05)
        G = nx.watts_strogatz_graph(n_nodes, k, p)
        
    elif graph_type == "Grid":
        # Create a grid graph
        dim = int(np.sqrt(n_nodes))
        G = nx.grid_2d_graph(dim, dim)
        # Relabel nodes to integers for consistency
        G = nx.convert_node_labels_to_integers(G)
        
    elif graph_type == "Cycle":
        G = nx.cycle_graph(n_nodes)
    
    # Add some random edge weights
    for u, v in G.edges():
        G[u][v]['weight'] = round(np.random.uniform(1, 10), 2)
    
    return G

def load_from_edge_list():
    """Load graph from an uploaded edge list file."""
    st.sidebar.subheader("Upload Edge List")
    
    uploaded_file = st.sidebar.file_uploader("Upload CSV or TXT file", type=["csv", "txt"])
    
    G = nx.Graph()
    
    if uploaded_file is not None:
        content = uploaded_file.getvalue().decode("utf-8")
        
        try:
            # Try to parse as CSV first
            data = pd.read_csv(StringIO(content), header=None)
            
            # Check for weighted edges
            if data.shape[1] >= 3:
                st.sidebar.info("Detected weighted edges")
                # Create weighted edges
                for _, row in data.iterrows():
                    G.add_edge(int(row[0]), int(row[1]), weight=float(row[2]))
            else:
                # Create unweighted edges
                for _, row in data.iterrows():
                    G.add_edge(int(row[0]), int(row[1]))
                
        except Exception as e:
            st.sidebar.error(f"Error parsing file: {e}")
            st.sidebar.info("Expected format: source_node,target_node[,weight]")
            return None
    
    return G

def load_from_adjacency_matrix():
    """Load graph from an uploaded adjacency matrix."""
    st.sidebar.subheader("Upload Adjacency Matrix")
    
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    
    G = nx.Graph()
    
    if uploaded_file is not None:
        try:
            # Read matrix
            adj_matrix = pd.read_csv(uploaded_file, header=None).values
            
            # Create graph from adjacency matrix
            n = adj_matrix.shape[0]
            
            for i in range(n):
                for j in range(i+1, n):  # Only upper triangle to avoid duplicates
                    if adj_matrix[i, j] != 0:
                        G.add_edge(i, j, weight=float(adj_matrix[i, j]))
                        
        except Exception as e:
            st.sidebar.error(f"Error parsing adjacency matrix: {e}")
            return None
    
    return G

def visualize_graph(G):
    """Visualize the graph using networkx and matplotlib."""
    st.subheader("Graph Visualization")
    
    # Get graph info
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    st.write(f"Graph has {n_nodes} nodes and {n_edges} edges.")
    
    # Visualization options
    col1, col2 = st.columns(2)
    
    with col1:
        layout_type = st.selectbox(
            "Layout",
            ["spring", "circular", "kamada_kawai", "random", "spectral", "shell"]
        )
    
    with col2:
        node_size = st.slider("Node Size", 50, 500, 300)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Get layout positions
    if layout_type == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout_type == "circular":
        pos = nx.circular_layout(G)
    elif layout_type == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout_type == "random":
        pos = nx.random_layout(G, seed=42)
    elif layout_type == "spectral":
        pos = nx.spectral_layout(G)
    elif layout_type == "shell":
        pos = nx.shell_layout(G)
    
    # Draw the graph
    nx.draw(
        G, pos, ax=ax, with_labels=True, node_size=node_size,
        node_color="skyblue", edge_color="gray", alpha=0.9,
        width=1.0, font_size=10
    )
    
    # Display the graph
    st.pyplot(fig)
    
    return pos

def get_neighborhood(G, node, depth=1):
    """Get neighborhood of a node up to a given depth."""
    neighbors = set([node])
    frontier = set([node])
    
    for _ in range(depth):
        new_frontier = set()
        for n in frontier:
            new_frontier.update(G.neighbors(n))
        new_frontier -= neighbors  # Remove already seen nodes
        neighbors.update(new_frontier)
        frontier = new_frontier
        
    return neighbors

def explore_neighborhood(G, pos=None):
    """Allow user to explore neighborhoods of nodes."""
    st.subheader("Neighborhood Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        node = st.selectbox(
            "Select a starting node:",
            sorted(list(G.nodes()))
        )
    
    with col2:
        depth = st.slider(
            "Neighborhood depth:",
            1, 5, 1,
            help="How many steps away from the selected node to include in the neighborhood"
        )
    
    if node is not None:
        # Get neighborhood
        neighborhood = get_neighborhood(G, node, depth)
        
        # Show neighborhood information
        st.write(f"Neighborhood of node {node} (depth={depth}) contains {len(neighborhood)} nodes.")
        
        # Calculate metrics for the subgraph
        subgraph = G.subgraph(neighborhood)
        
        # Get node metrics for the neighborhood
        try:
            degree_centrality = nx.degree_centrality(subgraph)
            betweenness_centrality = nx.betweenness_centrality(subgraph)
            closeness_centrality = nx.closeness_centrality(subgraph)
            
            # Create a DataFrame to display metrics
            metrics_df = pd.DataFrame({
                'Node': list(subgraph.nodes()),
                'Degree Centrality': [degree_centrality[n] for n in subgraph.nodes()],
                'Betweenness Centrality': [betweenness_centrality[n] for n in subgraph.nodes()],
                'Closeness Centrality': [closeness_centrality[n] for n in subgraph.nodes()]
            })
            
            metrics_df = metrics_df.sort_values('Degree Centrality', ascending=False)
            
            st.write("Node Centrality Metrics for This Neighborhood:")
            st.dataframe(metrics_df)
        except:
            st.warning("Unable to compute all centrality metrics for this neighborhood.")
        
        # Visualize the neighborhood
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Create a new position layout just for the subgraph
        subgraph_pos = nx.spring_layout(subgraph, seed=42)
        
        # Color nodes based on their distance from the source node
        node_colors = []
        for n in subgraph.nodes():
            if n == node:
                node_colors.append('red')  # Source node
            elif n in G.neighbors(node):
                node_colors.append('orange')  # Direct neighbors
            else:
                node_colors.append('skyblue')  # Further neighbors
        
        # Draw the neighborhood
        nx.draw(
            subgraph, subgraph_pos, ax=ax, with_labels=True,
            node_color=node_colors, edge_color="gray",
            node_size=300, alpha=0.9, width=1.0, font_size=10
        )
        
        st.write("Neighborhood Visualization:")
        st.pyplot(fig)
        
        # Show the shortest paths from the source node to other nodes in the neighborhood
        st.subheader("Shortest Paths Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target = st.selectbox(
                "Select a target node:",
                sorted(list(set(neighborhood) - {node}))
            )
        
        if target:
            try:
                # Find shortest path
                path = nx.shortest_path(G, source=node, target=target)
                
                st.write(f"Shortest path from {node} to {target}: {' → '.join(str(p) for p in path)}")
                
                # Visualize the shortest path
                fig, ax = plt.subplots(figsize=(10, 7))
                
                # Create edges for the path
                path_edges = list(zip(path, path[1:]))
                
                # Draw the full neighborhood
                nx.draw(
                    subgraph, subgraph_pos, ax=ax, with_labels=True,
                    node_color='lightgray', edge_color='lightgray',
                    node_size=300, alpha=0.5, width=1.0, font_size=10
                )
                
                # Highlight the path nodes
                nx.draw_networkx_nodes(
                    subgraph, subgraph_pos, ax=ax,
                    nodelist=path, node_color='orange',
                    node_size=300, alpha=1.0
                )
                
                # Highlight source and target
                nx.draw_networkx_nodes(
                    subgraph, subgraph_pos, ax=ax,
                    nodelist=[node], node_color='green',
                    node_size=400, alpha=1.0
                )
                
                nx.draw_networkx_nodes(
                    subgraph, subgraph_pos, ax=ax,
                    nodelist=[target], node_color='red',
                    node_size=400, alpha=1.0
                )
                
                # Highlight the path edges
                nx.draw_networkx_edges(
                    subgraph, subgraph_pos, ax=ax,
                    edgelist=path_edges, width=3.0,
                    edge_color='blue', alpha=1.0
                )
                
                st.write("Shortest Path Visualization:")
                st.pyplot(fig)
                
            except nx.NetworkXNoPath:
                st.error(f"No path exists between node {node} and node {target}.")

if __name__ == "__main__":
    main()