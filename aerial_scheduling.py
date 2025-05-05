import random
import numpy as np
import torch
import matplotlib.pyplot as plt
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

# --- Training function with better loss function ---
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
