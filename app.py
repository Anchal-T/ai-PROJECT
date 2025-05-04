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


class AerialScheduling:
    def __init__(self, num_vehicles, num_tasks, seed=42):
        random.seed(seed)
        self.graph = nx.Graph()
        self.num_vehicles = num_vehicles
        self.num_tasks = num_tasks
        self.vehicles = ["V" + str(i) for i in range(1, num_vehicles + 1)]
        self.tasks = ["T" + str(i) for i in range(1, num_tasks + 1)]
        self.solution = {}
        
        self.vehicle_features = {}
        self.task_features = {}
        
        for v in self.vehicles:
            # Vehicle features: fuel capacity, speed, payload capacity
            self.vehicle_features[v] = {
                'fuel': random.uniform(50, 100),  # Fuel level (50-100%)
                'speed': random.uniform(100, 200),  # Speed (100-200 km/h)
                'payload': random.uniform(5, 20)   # Payload capacity (5-20 units)
            }
        
        for t in self.tasks:
            # Task features: urgency, complexity, required_payload
            self.task_features[t] = {
                'urgency': random.uniform(1, 10),     # Urgency (1-10)
                'complexity': random.uniform(1, 10),  # Complexity (1-10)
                'required_payload': random.uniform(1, 15)  # Required payload (1-15 units)
            }

        # Add vehicles and tasks as nodes
        self.graph.add_nodes_from(self.vehicles, type="vehicle")
        self.graph.add_nodes_from(self.tasks, type="task")

        # Connect vehicles to tasks with weights that consider the features
        for v in self.vehicles:
            v_features = self.vehicle_features[v]
            for t in self.tasks:
                t_features = self.task_features[t]
                
                # Calculate a meaningful weight based on features
                # Higher urgency should reduce cost (better fit)
                # Higher complexity should increase cost if vehicle speed is low
                # Payload mismatch should increase cost
                
                urgency_factor = 10 / t_features['urgency']  # Inverse of urgency
                complexity_factor = t_features['complexity'] / (v_features['speed'] / 100)
                payload_mismatch = abs(v_features['payload'] - t_features['required_payload'])
                
                # Calculate the cost with some randomness
                base_cost = urgency_factor + complexity_factor + payload_mismatch
                weight = max(1, min(20, int(base_cost + random.uniform(-2, 2))))
                
                self.graph.add_edge(v, t, weight=weight)

    def get_graph_data(self):
        """Convert NetworkX graph to PyTorch Geometric format with rich features"""
        edge_index = []
        edge_attr = []
        node_features = []

        # Map node labels to indices
        node_index = {node: i for i, node in enumerate(self.graph.nodes())}

        for edge in self.graph.edges(data=True):
            v, t, attr = edge
            edge_index.append([node_index[v], node_index[t]])
            edge_index.append([node_index[t], node_index[v]])  # Bidirectional
            
            # Normalize edge weights to [0,1] range
            normalized_weight = attr["weight"] / 20.0  # Assuming max weight is 20
            edge_attr.append([normalized_weight])
            edge_attr.append([normalized_weight])

        # Create meaningful feature vectors
        for node in self.graph.nodes():
            if node.startswith("V"):  # Vehicle nodes
                v_features = self.vehicle_features[node]
                # Normalize features to [0,1] range
                features = [
                    v_features['fuel'] / 100.0,
                    v_features['speed'] / 200.0,
                    v_features['payload'] / 20.0,
                    1.0,  # Vehicle indicator
                    0.0   # Task indicator
                ]
                node_features.append(features)
            else:  # Task nodes
                t_features = self.task_features[node]
                # Normalize features to [0,1] range
                features = [
                    t_features['urgency'] / 10.0,
                    t_features['complexity'] / 10.0,
                    t_features['required_payload'] / 15.0,
                    0.0,  # Vehicle indicator
                    1.0   # Task indicator
                ]
                node_features.append(features)

        return Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float)
        )


# --- Define an enhanced GNN Model ---
class EnhancedGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EnhancedGNN, self).__init__()
        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Attention layer for better node relationships
        self.attention = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        
        # Output layer
        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Attention layer
        x = self.attention(x, edge_index)
        x = F.relu(x)
        
        # MLP for final processing
        x = self.lin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x

# --- Modified training function with better loss function ---
def train_gnn(graph_data, scheduler, num_epochs=200, learning_rate=0.005):
    model = EnhancedGNN(input_dim=5, hidden_dim=32, output_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Set model to training mode
    model.train()
    
    # Get proper node indices
    nodes_list = list(scheduler.graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}
    
    # Split vehicle and task nodes
    vehicle_nodes = [node for node in nodes_list if node.startswith("V")]
    task_nodes = [node for node in nodes_list if node.startswith("T")]
    
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        node_embeddings = model(graph_data)
        
        # Contrastive learning loss to encourage good assignments
        loss = 0.0
        
        # For each vehicle, create positive and negative examples
        for v in vehicle_nodes:
            v_idx = node_to_idx[v]
            v_embedding = node_embeddings[v_idx]
            
            # Find tasks with lowest and highest costs for this vehicle
            task_costs = [(t, scheduler.graph[v][t]["weight"]) for t in task_nodes]
            task_costs.sort(key=lambda x: x[1])
            
            # Use the lowest cost tasks as positive examples
            positive_tasks = task_costs[:2]  # 2 lowest cost tasks
            positive_loss = 0
            for t, _ in positive_tasks:
                t_idx = node_to_idx[t]
                t_embedding = node_embeddings[t_idx]
                # Pull positive examples closer in embedding space
                positive_loss += F.pairwise_distance(v_embedding.unsqueeze(0), t_embedding.unsqueeze(0))
            
            # Use the highest cost tasks as negative examples
            negative_tasks = task_costs[-2:]  # 2 highest cost tasks
            negative_loss = 0
            for t, _ in negative_tasks:
                t_idx = node_to_idx[t]
                t_embedding = node_embeddings[t_idx]
                # Push negative examples apart in embedding space (with margin)
                dist = F.pairwise_distance(v_embedding.unsqueeze(0), t_embedding.unsqueeze(0))
                negative_loss += torch.max(torch.tensor(0.0), 5.0 - dist)
            
            # Combine losses with appropriate weighting
            loss += positive_loss + 0.5 * negative_loss
            
        # Add regularization to prevent embeddings from collapsing
        embedding_norm = torch.norm(node_embeddings)
        loss += 0.001 * embedding_norm
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler_lr.step()
        
        if epoch % 20 == 0:
            # Evaluate current model performance
            model.eval()
            with torch.no_grad():
                current_assignments = optimized_assignment(model, graph_data, scheduler)
                current_cost = calculate_total_cost(current_assignments, scheduler)
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Current Cost = {current_cost}")
            model.train()
    
    return model

# --- Naive Task Assignment ---
def naive_assignment(scheduling):
    """Assign tasks to vehicles greedily (naïve method)"""
    assignment = {}
    assigned_tasks = set()

    for v in scheduling.vehicles:
        min_task = None
        min_cost = float('inf')

        for t in scheduling.tasks:
            if t not in assigned_tasks:
                cost = scheduling.graph[v][t]['weight']
                if cost < min_cost:
                    min_cost = cost
                    min_task = t

        if min_task:
            assignment[v] = min_task
            assigned_tasks.add(min_task)

    return assignment

# --- GNN-based Task Assignment ---
def optimized_assignment(model, graph_data, scheduler):
    """Use GNN embeddings to assign tasks to vehicles"""
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        node_embeddings = model(graph_data)
    
    # Get proper node indices
    nodes_list = list(scheduler.graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}
    
    # Split into vehicle and task nodes
    vehicle_nodes = [node for node in nodes_list if node.startswith("V")]
    task_nodes = [node for node in nodes_list if node.startswith("T")]
    
    # Calculate compatibility scores between all vehicles and tasks
    compatibility_matrix = np.zeros((len(vehicle_nodes), len(task_nodes)))
    
    for i, v in enumerate(vehicle_nodes):
        v_idx = node_to_idx[v]
        v_embedding = node_embeddings[v_idx]
        
        for j, t in enumerate(task_nodes):
            t_idx = node_to_idx[t]
            t_embedding = node_embeddings[t_idx]
            
            # Calculate compatibility score (negative distance)
            distance = F.pairwise_distance(v_embedding.unsqueeze(0), t_embedding.unsqueeze(0)).item()
            
            # Get the edge weight from the graph
            weight = scheduler.graph[v][t]["weight"]
            
            # Final score is a combination of embedding similarity and edge weight
            # Lower score means better assignment
            compatibility_matrix[i, j] = 0.7 * distance + 0.3 * weight
    
    # Hungarian algorithm for optimal assignment would be best here,
    # but we'll use a greedy approach for simplicity
    assignment = {}
    assigned_tasks = set()
    
    # Sort vehicles by their minimum compatibility score
    vehicle_indices = list(range(len(vehicle_nodes)))
    vehicle_indices.sort(key=lambda i: min(compatibility_matrix[i, j] for j in range(len(task_nodes))))
    
    for i in vehicle_indices:
        v = vehicle_nodes[i]
        best_task_idx = None
        best_score = float('inf')
        
        for j in range(len(task_nodes)):
            t = task_nodes[j]
            if t not in assigned_tasks:
                score = compatibility_matrix[i, j]
                if score < best_score:
                    best_score = score
                    best_task_idx = j
        
        if best_task_idx is not None:
            assignment[v] = task_nodes[best_task_idx]
            assigned_tasks.add(task_nodes[best_task_idx])
    
    return assignment

# --- Calculate Total Cost ---
def calculate_total_cost(assignment, scheduling):
    """Compute the sum of edge weights for a given assignment"""
    total_cost = sum(scheduling.graph[v][t]["weight"] for v, t in assignment.items())
    return total_cost

# --- Visualize Assignments ---
def visualize_assignments(scheduler, naive_schedule, optimized_schedule, run_number=None, improvement_percent=None):
    G = scheduler.graph.copy()
    
    # Create a subgraph with only the assigned edges
    naive_edges = [(v, t) for v, t in naive_schedule.items()]
    optimized_edges = [(v, t) for v, t in optimized_schedule.items()]
    
    plt.figure(figsize=(15, 6))
    
    # Plot naive assignment
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G, seed=42)  # Fixed layout for comparison
    
    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, 
                         nodelist=[n for n in G.nodes() if n.startswith('V')],
                         node_color='blue', 
                         node_size=700,
                         alpha=0.8)
    nx.draw_networkx_nodes(G, pos, 
                         nodelist=[n for n in G.nodes() if n.startswith('T')],
                         node_color='green', 
                         node_size=700,
                         alpha=0.8)
    
    # Draw assignment edges
    nx.draw_networkx_edges(G, pos, 
                         edgelist=naive_edges,
                         width=2, alpha=1, edge_color='red')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    
    # Edge labels
    edge_labels = {(v, t): G[v][t]['weight'] for v, t in naive_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    title = f"Naive Assignment (Cost: {calculate_total_cost(naive_schedule, scheduler)})"
    if run_number is not None:
        title = f"Run {run_number}: " + title
    plt.title(title)
    plt.axis('off')
    
    # Plot optimized assignment
    plt.subplot(1, 2, 2)
    
    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, 
                         nodelist=[n for n in G.nodes() if n.startswith('V')],
                         node_color='blue', 
                         node_size=700,
                         alpha=0.8)
    nx.draw_networkx_nodes(G, pos, 
                         nodelist=[n for n in G.nodes() if n.startswith('T')],
                         node_color='green', 
                         node_size=700,
                         alpha=0.8)
    
    # Draw assignment edges
    nx.draw_networkx_edges(G, pos, 
                         edgelist=optimized_edges,
                         width=2, alpha=1, edge_color='red')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif")
    
    # Edge labels
    edge_labels = {(v, t): G[v][t]['weight'] for v, t in optimized_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    title = f"GNN Optimized Assignment (Cost: {calculate_total_cost(optimized_schedule, scheduler)})"
    if improvement_percent is not None:
        title += f" - Improvement: {improvement_percent:.2f}%"
    plt.title(title)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('assignment_comparison.png')
    plt.show()
    
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


if __name__ == "__main__":
    main()