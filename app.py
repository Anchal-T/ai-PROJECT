import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from graph_neighborhood_search import GraphNeighborhoodSearch

def main():
    st.title("Graph Neural Neighborhood Search Visualization")
    st.markdown("""
    This application demonstrates a simplified version of the Graph Neural Neighborhood Search algorithm 
    for solving Vehicle Routing Problems (VRP). The implementation focuses on the graph neighborhood structure
    and greedy search algorithm without the machine learning components.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("Parameters")
    n_nodes = st.sidebar.slider("Number of nodes (including depot)", 5, 40, 40)
    k = st.sidebar.slider("K nearest neighbors", 2, 15, 5)
    vehicle_capacity = st.sidebar.slider("Vehicle capacity", 0.5, 3.0, 1.0, 0.1)
    
    # Initialize solver
    solver = GraphNeighborhoodSearch(n_nodes=n_nodes, k=k)
    solver.vehicle_capacity = vehicle_capacity
    
    # Generate random problem or use current if exists
    if 'solver' not in st.session_state or st.sidebar.button("Generate New Problem"):
        solver.generate_random_problem()
        st.session_state.solver = solver
    else:
        solver = st.session_state.solver
        # Update parameters if they've changed
        solver.n_nodes = n_nodes
        solver.k = k
        solver.vehicle_capacity = vehicle_capacity
        
    # Solve VRP
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Solve with Greedy Algorithm"):
            solver.greedy_vrp_solve()
            st.session_state.solved = True
    
    with col2:
        if st.button("Improve Solution (2-opt)") and 'solved' in st.session_state:
            solver.improve_routes()
    
    # Display problem details
    st.subheader("Problem Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nodes", n_nodes)
    with col2:
        st.metric("Neighbors (k)", k)
    with col3:
        st.metric("Vehicle Capacity", vehicle_capacity)
    
    # Show solution metrics if solved
    if hasattr(st.session_state, 'solved') and st.session_state.solved:
        st.subheader("Solution")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Number of Routes", len(solver.routes))
        with col2:
            st.metric("Total Distance", f"{solver.total_distance:.3f}")
        
        # Display routes
        st.subheader("Routes")
        for i, route in enumerate(solver.routes):
            route_str = " → ".join([str(node) for node in route])
            st.write(f"**Route {i+1}:** {route_str}")
            
            # Calculate route details
            route_distance = solver.calculate_route_distance(route)
            route_demand = sum(solver.demands[node] for node in route[1:-1])  # Exclude depot
            st.write(f"Distance: {route_distance:.3f}, Total Demand: {route_demand:.2f}")
    
    # Plot
    st.subheader("Visualization")
    
    # Show problem visualization (nodes and neighborhoods)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot nodes
    ax.scatter(solver.coords[1:, 0], solver.coords[1:, 1], c='blue', s=50, label='Customers')
    ax.scatter(solver.coords[0, 0], solver.coords[0, 1], c='red', s=100, marker='*', label='Depot')
    
    # Add demand labels
    for i in range(1, solver.n_nodes):
        ax.text(solver.coords[i, 0] + 0.01, solver.coords[i, 1] + 0.01, 
                f"{i}\n({solver.demands[i]:.1f})", fontsize=9)
    
    # Plot neighborhood connections
    if st.checkbox("Show Neighborhoods", value=True):
        for i in range(solver.n_nodes):
            for j in solver.nearest_neighbors[i]:
                ax.plot([solver.coords[i, 0], solver.coords[j, 0]], 
                        [solver.coords[i, 1], solver.coords[j, 1]], 
                        'gray', alpha=0.2, linestyle='--')
    
    # Plot routes if solved
    if hasattr(st.session_state, 'solved') and st.session_state.solved:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(solver.routes)))
        
        for i, route in enumerate(solver.routes):
            route_x = solver.coords[route, 0]
            route_y = solver.coords[route, 1]
            ax.plot(route_x, route_y, c=colors[i], linewidth=2, 
                    label=f"Route {i+1}")
    
    ax.set_title("VRP Problem Visualization")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig)
    
    # Show neighborhood information
    if st.checkbox("Show Neighborhood Details"):
        st.subheader("Neighborhood Details")
        node_idx = st.slider("Select Node", 0, n_nodes-1, 0)
        
        st.write(f"Node {node_idx} coordinates: ({solver.coords[node_idx][0]:.3f}, {solver.coords[node_idx][1]:.3f})")
        st.write(f"Demand: {solver.demands[node_idx]:.3f}")
        
        st.write("K-Nearest Neighbors:")
        neighbors = solver.nearest_neighbors[node_idx]
        
        # Create a small visualization of this node and its neighbors
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot nodes
        ax.scatter(solver.coords[:, 0], solver.coords[:, 1], c='lightgray', s=30)
        ax.scatter(solver.coords[node_idx, 0], solver.coords[node_idx, 1], c='red', s=100, label=f'Node {node_idx}')
        
        # Plot neighbors
        for i, neighbor in enumerate(neighbors):
            ax.scatter(solver.coords[neighbor, 0], solver.coords[neighbor, 1], c='blue', s=80, label=f'Neighbor {i+1}')
            # Draw connection
            ax.plot([solver.coords[node_idx, 0], solver.coords[neighbor, 0]], 
                    [solver.coords[node_idx, 1], solver.coords[neighbor, 1]], 
                    'blue', linestyle='-', alpha=0.6)
            
            # Add distance label
            mid_x = (solver.coords[node_idx, 0] + solver.coords[neighbor, 0]) / 2
            mid_y = (solver.coords[node_idx, 1] + solver.coords[neighbor, 1]) / 2
            dist = solver.distance_matrix[node_idx, neighbor]
            ax.text(mid_x, mid_y, f"{dist:.3f}", fontsize=8, bbox=dict(boxstyle="round", 
                                                                  fc="white", ec="gray", alpha=0.7))
        
        ax.set_title(f"Neighbors of Node {node_idx}")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        
        # Show neighbor details in table
        neighbor_data = []
        for i, neighbor in enumerate(neighbors):
            neighbor_data.append({
                "Neighbor Node": neighbor,
                "Distance": f"{solver.distance_matrix[node_idx, neighbor]:.4f}",
                "Demand": f"{solver.demands[neighbor]:.2f}",
                "Coordinates": f"({solver.coords[neighbor][0]:.3f}, {solver.coords[neighbor][1]:.3f})"
            })
        
        st.table(neighbor_data)

if __name__ == "__main__":
    main()