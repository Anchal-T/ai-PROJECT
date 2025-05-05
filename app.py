import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from graph_neighborhood_search import GraphNeighborhoodSearch
import random
import torch
import networkx as nx
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

# Import components from the new file
from aerial_scheduling import (
    AerialScheduling, 
    EnhancedGNN, 
    train_gnn, 
    naive_assignment, 
    optimized_assignment, 
    calculate_total_cost, 
    visualize_assignments
)

def display_solution_graph(scheduler, schedule, cost):
    G = scheduler.graph.copy()
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes() if n.startswith('V')], node_color='blue', node_size=700, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=[n for n in G.nodes() if n.startswith('T')], node_color='green', node_size=700, alpha=0.8)
    edges = [(v, t) for v, t in schedule.items()]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2, alpha=1, edge_color='red')
    nx.draw_networkx_labels(G, pos, font_size=12)
    labels = {(v, t): G[v][t]['weight'] for v, t in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)
    plt.title(f"Assignment (Cost: {cost})")
    plt.axis('off')
    st.pyplot(fig)

def show_scheduling():
    st.title("Aerial Task Scheduling with GNN")
    st.markdown("""
    This demonstration shows how Graph Neural Networks can optimize vehicle-task assignments
    compared to traditional greedy approaches.
    """)
    
    # Parameters
    st.sidebar.header("Scheduling Parameters")
    num_vehicles = st.sidebar.slider("Number of vehicles", 3, 10, 5)
    num_tasks = st.sidebar.slider("Number of tasks", 3, 10, 5)
    
    # Generate problem
    if "generated" not in st.session_state:
        st.session_state.generated = False
    if st.button("Generate Scheduling Problem"):
        random.seed(42)
        torch.manual_seed(42)
        scheduler = AerialScheduling(num_vehicles=num_vehicles, num_tasks=num_tasks)
        st.session_state.scheduler = scheduler
        st.session_state.graph_data = scheduler.get_graph_data()
        st.session_state.generated = True
        # reset solution flags
        st.session_state.show_naive = False
        st.session_state.show_optimized = False
    
    # Display problem after generation
    if st.session_state.generated:
        scheduler = st.session_state.scheduler
        st.subheader("Problem Graph")
        G = scheduler.graph.copy()
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        # Draw nodes and labels
        nx.draw(G, pos, with_labels=True, node_size=700, alpha=0.8,
                nodelist=[n for n in G.nodes() if n.startswith('V')], node_color='blue')
        nx.draw(G, pos, with_labels=False,
                nodelist=[n for n in G.nodes() if n.startswith('T')], node_color='green', node_size=700, alpha=0.8)
        st.pyplot(fig)

        # Show details
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Vehicle Details")
            vehicle_data = [{
                "Vehicle": v,
                "Fuel": f"{scheduler.vehicle_features[v]['fuel']:.1f}%",
                "Speed": f"{scheduler.vehicle_features[v]['speed']:.1f} km/h",
                "Payload": f"{scheduler.vehicle_features[v]['payload']:.1f} units"
            } for v in scheduler.vehicles]
            st.table(vehicle_data)
        with col2:
            st.subheader("Task Details")
            task_data = [{
                "Task": t,
                "Urgency": f"{scheduler.task_features[t]['urgency']:.1f}/10",
                "Complexity": f"{scheduler.task_features[t]['complexity']:.1f}/10",
                "Required Payload": f"{scheduler.task_features[t]['required_payload']:.1f} units"
            } for t in scheduler.tasks]
            st.table(task_data)

        # Solve buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Solve with Naive Assignment"):
                sched = st.session_state.scheduler
                st.session_state.naive_schedule = naive_assignment(sched)
                st.session_state.naive_cost = calculate_total_cost(st.session_state.naive_schedule, sched)
                st.session_state.show_naive = True
        with col2:
            if st.button("Solve with GNN"):
                sched = st.session_state.scheduler
                data = st.session_state.graph_data
                with st.spinner("Training GNN model - this may take a moment..."):
                    model = train_gnn(data, sched, num_epochs=150)
                    st.session_state.optimized_schedule = optimized_assignment(model, data, sched)
                    st.session_state.optimized_cost = calculate_total_cost(st.session_state.optimized_schedule, sched)
                st.session_state.show_optimized = True

        # Display naive solution if available
        if st.session_state.show_naive:
            st.success(f"Naive solution cost: {st.session_state.naive_cost}")
            st.subheader("Naive Assignment Solution")
            display_solution_graph(st.session_state.scheduler, st.session_state.naive_schedule, st.session_state.naive_cost)

        # Display GNN solution if available
        if st.session_state.show_optimized:
            st.success(f"GNN solution cost: {st.session_state.optimized_cost}")
            st.subheader("GNN Optimized Assignment")
            display_solution_graph(st.session_state.scheduler, st.session_state.optimized_schedule, st.session_state.optimized_cost)

        # Comparison
        if st.session_state.show_naive and st.session_state.show_optimized:
            st.subheader("Solution Comparison")
            n_cost = st.session_state.naive_cost
            o_cost = st.session_state.optimized_cost
            improvement = n_cost - o_cost
            pct = (improvement / n_cost * 100) if n_cost else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Naive Cost", f"{n_cost}")
            c2.metric("GNN Cost", f"{o_cost}")
            c3.metric("Improvement", f"{pct:.2f}%")
            st.markdown(f"The GNN approach is **{pct:.2f}%** better than naive.")

def main():
    app_mode = st.sidebar.radio(
        "Select Application",
        ["Graph Neural Neighborhood Search", "Aerial Task Scheduling"]
    )

    if app_mode == "Graph Neural Neighborhood Search":
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

        # Show neighborhood information (MOVED INSIDE THE IF BLOCK)
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
                             ax_detail.text(mid_x, mid_y, f"{dist:.3f}", fontsize=8, bbox=dict(boxstyle="round",fc="white", ec="gray", alpha=0.7))
                             
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

    elif app_mode == "Aerial Task Scheduling": # Handle the other mode
        show_scheduling() # Call the function for the other mode

if __name__ == "__main__":
    main()