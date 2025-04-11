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
    
    # Check if parameters have changed or if a new problem is requested
    generate_new = False
    if 'solver' not in st.session_state:
        generate_new = True
    elif st.session_state.n_nodes != n_nodes or st.session_state.k != k:
        generate_new = True
        # Clear previous solution state if parameters change
        if 'solved' in st.session_state:
            del st.session_state.solved
            
    if st.sidebar.button("Generate New Problem"):
        generate_new = True
        # Clear previous solution state on explicit regeneration
        if 'solved' in st.session_state:
            del st.session_state.solved

    # Initialize or update solver
    if generate_new:
        solver = GraphNeighborhoodSearch(n_nodes=n_nodes, k=k)
        solver.generate_random_problem()
        st.session_state.solver = solver
        st.session_state.n_nodes = n_nodes # Store current parameters
        st.session_state.k = k
        # Reset solved state when a new problem is generated
        if 'solved' in st.session_state:
             del st.session_state.solved
    else:
        solver = st.session_state.solver

    # Update vehicle capacity (can be changed without regenerating the whole problem)
    solver.vehicle_capacity = vehicle_capacity
            
    # Solve VRP
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Solve with Greedy Algorithm"):
            # Ensure the solver object has the latest parameters before solving
            # (This might be redundant now but good practice)
            solver.n_nodes = n_nodes 
            solver.k = k
            solver.vehicle_capacity = vehicle_capacity
            solver.greedy_vrp_solve()
            st.session_state.solved = True
            st.rerun() # Rerun to update the display immediately after solving
    
    with col2:
        # Only show improve button if a solution exists for the current problem
        if 'solved' in st.session_state and st.session_state.solved:
            if st.button("Improve Solution (2-opt)"):
                solver.improve_routes()
                st.rerun() # Rerun to update display after improvement
    
    # Display problem details
    st.subheader("Problem Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        # Use the actual number of nodes from the solver instance
        st.metric("Nodes", solver.n_nodes) 
    with col2:
        st.metric("Neighbors (k)", solver.k)
    with col3:
        st.metric("Vehicle Capacity", solver.vehicle_capacity)
    
    # Show solution metrics if solved
    # Check if 'solved' exists and is True for the current session state solver
    if 'solved' in st.session_state and st.session_state.solved and 'solver' in st.session_state and st.session_state.solver == solver:
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
            # Ensure demands array matches n_nodes before accessing
            if len(solver.demands) >= solver.n_nodes:
                 route_demand = sum(solver.demands[node] for node in route[1:-1]) # Exclude depot
                 st.write(f"Distance: {route_distance:.3f}, Total Demand: {route_demand:.2f}")
            else:
                 st.write(f"Distance: {route_distance:.3f}, Demand calculation skipped (data mismatch)")

    
    # Plot
    st.subheader("Visualization")
    
    # Ensure coordinates and demands arrays are valid before plotting
    if len(solver.coords) >= solver.n_nodes and len(solver.demands) >= solver.n_nodes:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot nodes
        ax.scatter(solver.coords[1:solver.n_nodes, 0], solver.coords[1:solver.n_nodes, 1], c='blue', s=50, label='Customers')
        ax.scatter(solver.coords[0, 0], solver.coords[0, 1], c='red', s=100, marker='*', label='Depot')
        
        # Add demand labels
        for i in range(1, solver.n_nodes):
            ax.text(solver.coords[i, 0] + 0.01, solver.coords[i, 1] + 0.01, 
                    f"{i}\n({solver.demands[i]:.1f})", fontsize=9)
        
        # Plot neighborhood connections
        if st.checkbox("Show Neighborhoods", value=True):
             # Ensure nearest_neighbors array is valid
             if len(solver.nearest_neighbors) >= solver.n_nodes:
                 for i in range(solver.n_nodes):
                     # Check if neighbors list for node i exists and is valid
                     if i < len(solver.nearest_neighbors) and solver.nearest_neighbors[i] is not None:
                         for j in solver.nearest_neighbors[i]:
                             # Check if neighbor index j is valid
                             if j < len(solver.coords):
                                 ax.plot([solver.coords[i, 0], solver.coords[j, 0]], 
                                         [solver.coords[i, 1], solver.coords[j, 1]], 
                                         'gray', alpha=0.2, linestyle='--')
        
        # Plot routes if solved
        if 'solved' in st.session_state and st.session_state.solved and 'solver' in st.session_state and st.session_state.solver == solver:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(solver.routes)))
            
            for i, route in enumerate(solver.routes):
                 # Ensure all nodes in the route are valid indices for coords
                 if all(node < len(solver.coords) for node in route):
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
    else:
        st.warning("Problem data (coordinates/demands) is inconsistent with the number of nodes. Please generate a new problem.")

    
    # Show neighborhood information
    if st.checkbox("Show Neighborhood Details"):
        st.subheader("Neighborhood Details")
        # Use the actual number of nodes from the solver for the slider limit
        node_idx = st.slider("Select Node", 0, solver.n_nodes - 1, 0) 
        
        # Add checks before accessing solver data based on node_idx
        if node_idx < len(solver.coords) and node_idx < len(solver.demands) and node_idx < len(solver.nearest_neighbors):
            st.write(f"Node {node_idx} coordinates: ({solver.coords[node_idx][0]:.3f}, {solver.coords[node_idx][1]:.3f})")
            st.write(f"Demand: {solver.demands[node_idx]:.3f}")
            
            st.write("K-Nearest Neighbors:")
            neighbors = solver.nearest_neighbors[node_idx]
            
            # Create a small visualization of this node and its neighbors
            fig_detail, ax_detail = plt.subplots(figsize=(6, 6))
            
            # Plot nodes (ensure coords is valid)
            if len(solver.coords) > 0:
                 ax_detail.scatter(solver.coords[:, 0], solver.coords[:, 1], c='lightgray', s=30)
                 ax_detail.scatter(solver.coords[node_idx, 0], solver.coords[node_idx, 1], c='red', s=100, label=f'Node {node_idx}')
            
            # Plot neighbors
            neighbor_data = []
            if neighbors is not None:
                for i, neighbor in enumerate(neighbors):
                     # Check if neighbor index and distance matrix are valid
                     if neighbor < len(solver.coords) and node_idx < solver.distance_matrix.shape[0] and neighbor < solver.distance_matrix.shape[1]:
                         ax_detail.scatter(solver.coords[neighbor, 0], solver.coords[neighbor, 1], c='blue', s=80, label=f'Neighbor {i+1}')
                         # Draw connection
                         ax_detail.plot([solver.coords[node_idx, 0], solver.coords[neighbor, 0]], 
                                 [solver.coords[node_idx, 1], solver.coords[neighbor, 1]], 
                                 'blue', linestyle='-', alpha=0.6)
                         
                         # Add distance label
                         mid_x = (solver.coords[node_idx, 0] + solver.coords[neighbor, 0]) / 2
                         mid_y = (solver.coords[node_idx, 1] + solver.coords[neighbor, 1]) / 2
                         dist = solver.distance_matrix[node_idx, neighbor]
                         ax_detail.text(mid_x, mid_y, f"{dist:.3f}", fontsize=8, bbox=dict(boxstyle="round", 
                                                                               fc="white", ec="gray", alpha=0.7))
                         
                         # Prepare data for table (check demands array validity)
                         if neighbor < len(solver.demands):
                             neighbor_data.append({
                                 "Neighbor Node": neighbor,
                                 "Distance": f"{dist:.4f}",
                                 "Demand": f"{solver.demands[neighbor]:.2f}",
                                 "Coordinates": f"({solver.coords[neighbor][0]:.3f}, {solver.coords[neighbor][1]:.3f})"
                             })
            
            ax_detail.set_title(f"Neighbors of Node {node_idx}")
            ax_detail.legend()
            ax_detail.set_xlim(0, 1)
            ax_detail.set_ylim(0, 1)
            ax_detail.grid(True, linestyle='--', alpha=0.7)
            
            st.pyplot(fig_detail)
            
            # Show neighbor details in table
            if neighbor_data:
                st.table(neighbor_data)
        else:
             st.warning(f"Selected node index {node_idx} is out of bounds for the current problem data. Please generate a new problem or select a valid node.")


if __name__ == "__main__":
    main()