import numpy as np
import pandas as pd
import networkx as nx

def create_sample_edge_list(filename="sample_edge_list.csv", n_nodes=20, p=0.2):
    """
    Create a sample edge list file for testing the graph neighborhood app.
    
    Parameters:
    - filename: Name of the output file
    - n_nodes: Number of nodes in the graph
    - p: Probability of edge creation (for Erdős-Rényi graph)
    """
    # Create a random graph
    G = nx.erdos_renyi_graph(n_nodes, p, seed=42)
    
    # Add random weights to edges
    for u, v in G.edges():
        G[u][v]['weight'] = round(np.random.uniform(1, 10), 2)
    
    # Create a dataframe with the edge list
    edge_list = []
    for u, v, data in G.edges(data=True):
        edge_list.append([u, v, data['weight']])
    
    # Convert to dataframe and save
    df = pd.DataFrame(edge_list, columns=['source', 'target', 'weight'])
    df.to_csv(filename, index=False, header=False)
    
    print(f"Created sample edge list file: {filename}")
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G

def create_sample_adjacency_matrix(filename="sample_adjacency_matrix.csv", n_nodes=20, p=0.2):
    """
    Create a sample adjacency matrix file for testing the graph neighborhood app.
    
    Parameters:
    - filename: Name of the output file
    - n_nodes: Number of nodes in the graph
    - p: Probability of edge creation (for Erdős-Rényi graph)
    """
    # Create a random graph
    G = nx.erdos_renyi_graph(n_nodes, p, seed=42)
    
    # Add random weights to edges
    for u, v in G.edges():
        G[u][v]['weight'] = round(np.random.uniform(1, 10), 2)
    
    # Get adjacency matrix
    adj_matrix = nx.to_numpy_array(G)
    
    # Save to CSV
    pd.DataFrame(adj_matrix).to_csv(filename, index=False, header=False)
    
    print(f"Created sample adjacency matrix file: {filename}")
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G

if __name__ == "__main__":
    # Create sample data files
    create_sample_edge_list()
    create_sample_adjacency_matrix()