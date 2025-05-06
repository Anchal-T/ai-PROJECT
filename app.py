import torch.nn as nn
from torch.nn import MultiheadAttention
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
import time
import sys
import os

from aerial_scheduling import (
    AerialScheduling, 
    EnhancedGNN, 
    train_gnn, 
    naive_assignment, 
    optimized_assignment, 
    calculate_total_cost, 
    visualize_assignments
)

# Import the GENIS model
from genis_rl_model import (
    SimpleGENISAgent, 
    SimpleVRPEnvironment,
    NUM_ACTIONS, 
    HIDDEN_DIM, 
    OUTPUT_DIM, 
    GAMMA, 
    EPSILON_START, 
    EPSILON_END, 
    EPSILON_DECAY,
    BATCH_SIZE,
    REPLAY_BUFFER_CAPACITY
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

def build_initial_solution(customers, depots, N_i):
    s0 = {i: {j: [depots[i], depots[i]] for j in range(N_i[i])} for i in depots}
    RC = set(customers)
    while RC:
        regrets = {}
        for ck in RC:
            deltas = []
            for i in s0:
                for j in s0[i]:
                    d = insertion_cost(s0[i][j], ck)
                    deltas.append((d, i, j))
            deltas.sort(key=lambda x: x[0])
            if len(deltas) > 1:
                rv = deltas[1][0] - deltas[0][0]
            else:
                rv = deltas[0][0]
            regrets[ck] = (rv, deltas[0][1], deltas[0][2])
        # pick ck* with max regret
        ck_star, (rv, i_star, j_star) = max(regrets.items(), key=lambda x: x[1][0])
        s0[i_star][j_star].insert(-1, ck_star)
        RC.remove(ck_star)
    return s0

class DualGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, heads=4):
        super().__init__()
        self.gcn1 = GCNConv(in_dim, hid_dim)
        self.gcn2 = GCNConv(in_dim, hid_dim)
        self.lin1 = nn.Linear(hid_dim, out_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)
        self.attn = MultiheadAttention(embed_dim=out_dim, num_heads=heads)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x, edge_index_dis, edge_index_sol):
        y1 = torch.relu(self.gcn1(x, edge_index_dis))
        y2 = torch.relu(self.gcn2(x, edge_index_sol))
        y1 = self.lin1(y1)
        y2 = self.lin2(y2)
        y = torch.stack([y1, y2], dim=0)
        y, _ = self.attn(y, y, y)
        fused = y.sum(dim=0)
        return self.mlp(fused)

def local_search(current_solution, operator):
    best = current_solution
    best_cost = cost(best)
    for neighbor in operator(current_solution):
        c = cost(neighbor)
        if c < best_cost:
            best, best_cost = neighbor, c
    return best

def reward_shaping(batch, init_cost):
    omegas = [abs(cost(s) - init_cost) / init_cost for s, a, r, s_next in batch]
    total_ω = sum(omegas) + 1e-8
    shaped = []
    for (s, a, r, s_next), ω in zip(batch, omegas):
        shaped.append((s, a, ω * r / total_ω, s_next))
    rs = [r_hat for _,_,r_hat,_ in shaped]
    r_min, r_max = min(rs), max(rs)
    normed = []
    for s, a, r_hat, s_next in shaped:
        if r_max > r_min:
            r_hat = (r_hat - r_min) / (r_max - r_min)
        normed.append((s, a, r_hat, s_next))
    return normed

def update_policy(replay_buffer, init_cost, gamma, q_net, target_q, optimizer, batch_size):
    batch = random.sample(replay_buffer, batch_size)
    shaped = reward_shaping(batch, init_cost)
    losses = []
    for s, a, r, s_next in shaped:
        φ_s    = q_net(s.x, s.edge_index_dis, s.edge_index_sol)
        φ_next = target_q(s_next.x, s_next.edge_index_dis, s_next.edge_index_sol)
        q_val  = φ_s[a]
        y      = r + gamma * φ_next.max()
        losses.append((q_val - y).pow(2))
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # soft update:
    for p, p_t in zip(q_net.parameters(), target_q.parameters()):
        p_t.data.mul_(0.95).add_(0.05, p.data)
        
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
    
    st.sidebar.header("Scheduling Parameters")
    num_vehicles = st.sidebar.slider("Number of vehicles", 3, 10, 5)
    num_tasks = st.sidebar.slider("Number of tasks", 3, 10, 5)
    
    if "generated" not in st.session_state:
        st.session_state.generated = False
    if st.button("Generate Scheduling Problem"):
        random.seed(42)
        torch.manual_seed(42)
        scheduler = AerialScheduling(num_vehicles=num_vehicles, num_tasks=num_tasks)
        st.session_state.scheduler = scheduler
        st.session_state.graph_data = scheduler.get_graph_data()
        st.session_state.generated = True
        st.session_state.show_naive = False
        st.session_state.show_optimized = False
    
    if st.session_state.generated:
        scheduler = st.session_state.scheduler
        st.subheader("Problem Graph")
        G = scheduler.graph.copy()
        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_size=700, alpha=0.8,
                nodelist=[n for n in G.nodes() if n.startswith('V')], node_color='blue')
        nx.draw(G, pos, with_labels=False,
                nodelist=[n for n in G.nodes() if n.startswith('T')], node_color='green', node_size=700, alpha=0.8)
        st.pyplot(fig)

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

        if st.session_state.show_naive:
            st.success(f"Naive solution cost: {st.session_state.naive_cost}")
            st.subheader("Naive Assignment Solution")
            display_solution_graph(st.session_state.scheduler, st.session_state.naive_schedule, st.session_state.naive_cost)

        if st.session_state.show_optimized:
            st.success(f"GNN solution cost: {st.session_state.optimized_cost}")
            st.subheader("GNN Optimized Assignment")
            display_solution_graph(st.session_state.scheduler, st.session_state.optimized_schedule, st.session_state.optimized_cost)

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
            
# Function to train the model with progress feedback
def train_model_with_progress(num_customers, num_episodes, max_steps_per_episode, 
                             learning_rate, gamma, hidden_dim, epsilon_start, 
                             epsilon_end, epsilon_decay, batch_size, buffer_capacity,
                             grid_size=100):
    # Initialize environment and agent
    env = SimpleVRPEnvironment(num_customers=num_customers, grid_size=grid_size)
    agent = SimpleGENISAgent(
        num_actions=NUM_ACTIONS,
        hidden_dim=hidden_dim,
        output_dim=OUTPUT_DIM,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay
    )
    
    # Initialize tracking variables
    best_cost = float('inf')
    costs_over_time = []
    best_solution = None
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward_sum = 0.0
        
        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            loss = agent.update(batch_size=batch_size)
            
            state = next_state
            episode_reward_sum += reward
            
            if done:
                break
        
        # Update exploration rate
        agent.update_epsilon()
        
        # Track best solution
        final_cost = state[2]  # Cost is the 3rd element of the state tuple
        costs_over_time.append(final_cost)
        
        if final_cost < best_cost:
            best_cost = final_cost
            best_solution = env.tour.copy()
            best_coords = env.coords.clone() if isinstance(env.coords, torch.Tensor) else torch.tensor(env.coords)
        
        # Update progress bar and status text
        progress = (episode + 1) / num_episodes
        progress_bar.progress(progress)
        
        # Update status every few episodes
        if (episode + 1) % max(1, (num_episodes // 10)) == 0 or episode == 0:
            status_text.text(f"Episode {episode+1}/{num_episodes} | Current Cost: {final_cost:.2f} | Best Cost: {best_cost:.2f} | Epsilon: {agent.epsilon:.3f}")
    
    # Clear progress bar and status text
    progress_bar.empty()
    status_text.empty()
    
    # Store the best solution in the environment for visualization
    if best_solution:
        env.tour = best_solution
        env.coords = best_coords
        env.current_cost = best_cost
    
    return agent, env, costs_over_time

# Function to visualize the solution
def visualize_solution(env, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    coords_np = env.coords.numpy() if isinstance(env.coords, torch.Tensor) else env.coords
    tour = env.tour
    
    # Plot depot
    ax.scatter(coords_np[0, 0], coords_np[0, 1], c='red', s=200, marker='s', label='Depot (0)', zorder=3)
    
    # Plot customers
    if len(coords_np) > 1:
        ax.scatter(coords_np[1:, 0], coords_np[1:, 1], c='blue', s=50, label='Customers', zorder=2)
    
    # Plot the tour lines
    if len(tour) > 1:
        tour_coords = coords_np[tour]
        ax.plot(tour_coords[:, 0], tour_coords[:, 1], 'k-', linewidth=1.0, alpha=0.7, zorder=1)
    
    # Annotate customers (nodes 1 to N)
    for i in range(1, len(coords_np)):
        ax.annotate(f"{i}", (coords_np[i, 0], coords_np[i, 1]),
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_title(f"VRP Solution - Cost: {env.current_cost:.2f}")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')
    
    return ax

# Function to plot the learning curve
def plot_learning_curve(costs, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(costs, 'b-')
    ax.set_title('Learning Curve (Solution Cost per Episode)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Solution Cost')
    ax.grid(True)
    
    return ax

# Set random seeds for reproducibility
@st.cache_resource
def set_seeds(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return True

def main():
    app_mode = st.sidebar.radio(
        "Select Application",
        ["Graph Neural Neighborhood Search", "Aerial Task Scheduling", "GENIS RL for VRP"]
    )

    if app_mode == "Graph Neural Neighborhood Search":
        st.title("Graph Neural Neighborhood Search Visualization")
        st.markdown(""" 
        This application demonstrates a simplified version of the Graph Neural Neighborhood Search algorithm 
        for solving Vehicle Routing Problems (VRP). The implementation focuses on the graph neighborhood structure
        and greedy search algorithm without the machine learning components.
        """)

        st.sidebar.header("Parameters")
        n_nodes = st.sidebar.slider("Number of nodes (including depot)", 5, 40, 40)
        k = st.sidebar.slider("K nearest neighbors", 2, 15, 5)
        vehicle_capacity = st.sidebar.slider("Vehicle capacity", 0.5, 3.0, 1.0, 0.1)
                
        generate_new = False
        if 'solver' not in st.session_state:
            generate_new = True
        elif st.session_state.n_nodes != n_nodes or st.session_state.k != k:
            generate_new = True
            if 'solved' in st.session_state:
                del st.session_state.solved
            if 'distance_history' in st.session_state:
                del st.session_state.distance_history
                
        if st.sidebar.button("Generate New Problem"):
            generate_new = True
            if 'solved' in st.session_state:
                del st.session_state.solved
            if 'distance_history' in st.session_state:
                del st.session_state.distance_history

        if generate_new:
            solver = GraphNeighborhoodSearch(n_nodes=n_nodes, k=k)
            solver.generate_random_problem()
            st.session_state.solver = solver
            st.session_state.n_nodes = n_nodes
            st.session_state.k = k
            if 'solved' in st.session_state:
                del st.session_state.solved
            st.session_state.distance_history = []
        else:
            solver = st.session_state.solver
            if 'distance_history' not in st.session_state:
                 st.session_state.distance_history = []

        solver.vehicle_capacity = vehicle_capacity
                
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Solve with Greedy Algorithm"):
                solver.n_nodes = n_nodes 
                solver.k = k
                solver.vehicle_capacity = vehicle_capacity
                solver.greedy_vrp_solve()
                st.session_state.solved = True
                st.session_state.distance_history = [solver.total_distance] 
                st.rerun()
        
        with col2:
            if 'solved' in st.session_state and st.session_state.solved:
                if st.button("Improve Solution (2-opt)"):
                    solver.improve_routes()
                    st.session_state.distance_history.append(solver.total_distance)
                    st.rerun()
        
        st.subheader("Problem Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nodes", solver.n_nodes) 
        with col2:
            st.metric("Neighbors (k)", solver.k)
        with col3:
            st.metric("Vehicle Capacity", solver.vehicle_capacity)
        
        if 'solved' in st.session_state and st.session_state.solved and 'solver' in st.session_state and st.session_state.solver == solver:
            st.subheader("Solution")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Routes", len(solver.routes))
            with col2:
                st.metric("Total Distance", f"{solver.total_distance:.3f}")
            
            st.subheader("Routes")
            for i, route in enumerate(solver.routes):
                route_str = " → ".join([str(node) for node in route])
                st.write(f"**Route {i+1}:** {route_str}")
                
                route_distance = solver.calculate_route_distance(route)
                if len(solver.demands) >= solver.n_nodes:
                    route_demand = sum(solver.demands[node] for node in route[1:-1])
                    st.write(f"Distance: {route_distance:.3f}, Total Demand: {route_demand:.2f}")
                else:
                    st.write(f"Distance: {route_distance:.3f}, Demand calculation skipped (data mismatch)")

            if 'distance_history' in st.session_state and len(st.session_state.distance_history) > 1:
                st.subheader("Total Distance Improvement")
                history_data = np.array(st.session_state.distance_history)
                labels = ["Greedy"] + [f"{i}-opt" for i in range(1, len(history_data))]
                
                fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
                
                ax_hist.set_facecolor('white')
                fig_hist.patch.set_facecolor('white')
                
                ax_hist.grid(False)
                
                ax_hist.spines['top'].set_visible(False)
                ax_hist.spines['right'].set_visible(False)
                ax_hist.spines['left'].set_color('#dddddd')
                ax_hist.spines['bottom'].set_color('#dddddd')
                
                ax_hist.plot(labels, history_data, marker='o', linestyle='-', linewidth=2.5, 
                            color='#1f77b4', markersize=8, markerfacecolor='white', 
                            markeredgewidth=2, markeredgecolor='#1f77b4')
                
                for i, v in enumerate(history_data):
                    ax_hist.text(i, v + 0.0003, f"{v:.3f}", ha='center', va='bottom', 
                                fontweight='bold', fontsize=10)
                
                ax_hist.set_xlabel("Improvement Step", fontsize=12, fontweight='bold')
                ax_hist.set_ylabel("Total Distance", fontsize=12, fontweight='bold')
                ax_hist.set_title("Total Distance vs. Improvement Steps", 
                                fontsize=16, fontweight='bold', pad=20)
                
                ax_hist.tick_params(axis='both', which='major', labelsize=10)
                
                improvement = history_data[0] - history_data[-1]
                percent_improvement = (improvement / history_data[0]) * 100
                ax_hist.annotate(f"Improvement: {improvement:.3f} ({percent_improvement:.1f}%)",
                                xy=(0.5, 0.05), xycoords='axes fraction',
                                ha='center', va='bottom',
                                bbox=dict(boxstyle="round,pad=0.3", fc="#e9f7ef", ec="#1f77b4", alpha=0.7))
                
                plt.tight_layout()
                
                st.pyplot(fig_hist)
        
        st.subheader("Visualization")
        
        if len(solver.coords) >= solver.n_nodes and len(solver.demands) >= solver.n_nodes:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            ax.scatter(solver.coords[1:solver.n_nodes, 0], solver.coords[1:solver.n_nodes, 1], c='blue', s=50, label='Customers')
            ax.scatter(solver.coords[0, 0], solver.coords[0, 1], c='red', s=100, marker='*', label='Depot')
            
            for i in range(1, solver.n_nodes):
                ax.text(solver.coords[i, 0] + 0.01, solver.coords[i, 1] + 0.01, 
                        f"{i}\n({solver.demands[i]:.1f})", fontsize=9)
            
            if st.checkbox("Show Neighborhoods", value=True):
                if len(solver.nearest_neighbors) >= solver.n_nodes:
                    for i in range(solver.n_nodes):
                        if i < len(solver.nearest_neighbors) and solver.nearest_neighbors[i] is not None:
                            for j in solver.nearest_neighbors[i]:
                                if j < len(solver.coords):
                                    ax.plot([solver.coords[i, 0], solver.coords[j, 0]], 
                                            [solver.coords[i, 1], solver.coords[j, 1]], 
                                            'gray', alpha=0.2, linestyle='--')
            
            if 'solved' in st.session_state and st.session_state.solved and 'solver' in st.session_state and st.session_state.solver == solver:
                colors = plt.cm.rainbow(np.linspace(0, 1, len(solver.routes)))
                
                for i, route in enumerate(solver.routes):
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

        if st.checkbox("Show Neighborhood Details"):
            st.subheader("Neighborhood Details")
            node_idx = st.slider("Select Node", 0, solver.n_nodes - 1, 0) 
            
            if node_idx < len(solver.coords) and node_idx < len(solver.demands) and node_idx < len(solver.nearest_neighbors):
                st.write(f"Node {node_idx} coordinates: ({solver.coords[node_idx][0]:.3f}, {solver.coords[node_idx][1]:.3f})")
                st.write(f"Demand: {solver.demands[node_idx]:.3f}")
                
                st.write("K-Nearest Neighbors:")
                neighbors = solver.nearest_neighbors[node_idx]
                
                fig_detail, ax_detail = plt.subplots(figsize=(6, 6))
                
                if len(solver.coords) > 0:
                     ax_detail.scatter(solver.coords[:, 0], solver.coords[:, 1], c='lightgray', s=30)
                     ax_detail.scatter(solver.coords[node_idx, 0], solver.coords[node_idx, 1], c='red', s=100, label=f'Node {node_idx}')
                
                neighbor_data = []
                if neighbors is not None:
                    for i, neighbor in enumerate(neighbors):
                         if neighbor < len(solver.coords) and node_idx < solver.distance_matrix.shape[0] and neighbor < solver.distance_matrix.shape[1]:
                             ax_detail.scatter(solver.coords[neighbor, 0], solver.coords[neighbor, 1], c='blue', s=80, label=f'Neighbor {i+1}')
                             ax_detail.plot([solver.coords[node_idx, 0], solver.coords[neighbor, 0]], 
                                     [solver.coords[node_idx, 1], solver.coords[neighbor, 1]], 
                                     'blue', linestyle='-', alpha=0.6)
                             
                             mid_x = (solver.coords[node_idx, 0] + solver.coords[neighbor, 0]) / 2
                             mid_y = (solver.coords[node_idx, 1] + solver.coords[neighbor, 1]) / 2
                             dist = solver.distance_matrix[node_idx, neighbor]
                             ax_detail.text(mid_x, mid_y, f"{dist:.3f}", fontsize=8, bbox=dict(boxstyle="round",fc="white", ec="gray", alpha=0.7))
                             
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
                
                if neighbor_data:
                    st.table(neighbor_data)
            else:
                 st.warning(f"Selected node index {node_idx} is out of bounds for the current problem data. Please generate a new problem or select a valid node.")

    elif app_mode == "Aerial Task Scheduling":
        show_scheduling()
        
    elif app_mode == "GENIS RL for VRP":
        set_seeds()
    
        # App title and intro
        st.title("🚚 GENIS RL for Vehicle Routing Problem")
        st.write("""
        This app demonstrates the GENIS (Graph Embedding-based Neural Improvement Search) reinforcement learning model 
        applied to the Vehicle Routing Problem (VRP). The model uses a Graph Neural Network to learn 
        which search operators to apply to improve VRP solutions.
        """)
        
        # Sidebar for parameters
        st.sidebar.title("Parameters")
        
        # Problem parameters
        st.sidebar.header("Problem Settings")
        num_customers = st.sidebar.slider("Number of Customers", 5, 50, 20)
        grid_size = st.sidebar.slider("Grid Size", 50, 200, 100)
        
        # Training parameters
        st.sidebar.header("Training Settings")
        num_episodes = st.sidebar.slider("Number of Episodes", 10, 500, 50)
        max_steps = st.sidebar.slider("Max Steps per Episode", 50, 300, 150)
        
        # Agent parameters
        st.sidebar.header("Agent Settings")
        learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
        gamma = st.sidebar.slider("Discount Factor (Gamma)", 0.8, 0.999, GAMMA, format="%.3f")
        hidden_dim = st.sidebar.slider("Hidden Dimension", 8, 64, HIDDEN_DIM)
        
        # Exploration parameters
        st.sidebar.header("Exploration Settings")
        epsilon_start = st.sidebar.slider("Epsilon Start", 0.5, 1.0, EPSILON_START, format="%.2f")
        epsilon_end = st.sidebar.slider("Epsilon End", 0.01, 0.2, EPSILON_END, format="%.2f")
        epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.9, 0.999, EPSILON_DECAY, format="%.3f")
        
        # Memory parameters
        st.sidebar.header("Memory Settings")
        batch_size = st.sidebar.slider("Batch Size", 8, 128, BATCH_SIZE)
        buffer_capacity = st.sidebar.slider("Replay Buffer Capacity", 500, 10000, REPLAY_BUFFER_CAPACITY)
        
        # Tabs
        tab1, tab2 = st.tabs(["Training", "About GENIS"])
        
        with tab1:
            # Training section
            st.header("Train the Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Controls")
                train_button = st.button("Train Model", key="train_button", use_container_width=True)
            
            # Initialize session state variables
            if 'agent' not in st.session_state:
                st.session_state.agent = None
                st.session_state.env = None
                st.session_state.costs = None
                st.session_state.trained = False
            
            # Run training if button is clicked
            if train_button:
                with st.spinner('Training in progress...'):
                    agent, env, costs = train_model_with_progress(
                        num_customers=num_customers,
                        num_episodes=num_episodes,
                        max_steps_per_episode=max_steps,
                        learning_rate=learning_rate,
                        gamma=gamma,
                        hidden_dim=hidden_dim,
                        epsilon_start=epsilon_start,
                        epsilon_end=epsilon_end,
                        epsilon_decay=epsilon_decay,
                        batch_size=batch_size,
                        buffer_capacity=buffer_capacity,
                        grid_size=grid_size
                    )
                    
                    # Store results in session state
                    st.session_state.agent = agent
                    st.session_state.env = env
                    st.session_state.costs = costs
                    st.session_state.trained = True
                    
                    st.success(f"Training completed! Best solution cost: {env.current_cost:.2f}")
            
            # Display results if training has been completed
            if st.session_state.trained:
                st.subheader("Training Results")
                
                # Create figure with two subplots for solution and learning curve
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                
                # Visualize solution
                visualize_solution(st.session_state.env, ax=ax1)
                
                # Plot learning curve
                plot_learning_curve(st.session_state.costs, ax=ax2)
                
                # Adjust layout
                plt.tight_layout()
                
                # Display the plot
                st.pyplot(fig)
                
                # Display solution details
                st.subheader("Solution Details")
                
                # Show tour information
                st.write("**Tour:**")
                tour_str = " → ".join([str(node) for node in st.session_state.env.tour])
                st.code(tour_str)

                st.write("**Solution Cost:**")
                st.info(f"{st.session_state.env.current_cost:.2f}")
        
        with tab2:
            # About GENIS section
            st.header("About GENIS")
            st.write("""
            ## Graph Embedding Neural Improvement Search
            
            GENIS (Graph Embedding Neural Improvement Search) is a reinforcement learning approach 
            that combines graph neural networks with reinforcement learning to solve combinatorial 
            optimization problems like the Vehicle Routing Problem (VRP).
            
            ### How it works:
            
            1. **Graph Representation**: The VRP is represented as a graph where nodes are customers 
            and edges represent the routes between them.
            
            2. **Graph Neural Network**: A GNN processes the graph structure to create embeddings 
            that capture the problem's structural information.
            
            3. **Reinforcement Learning**: An agent learns which search operators (like 2-opt, swap, etc.) 
            to apply to improve the solution.
            
            4. **Search Operators**: Four basic operators are implemented:
            - **Swap**: Exchange the positions of two customers in the tour
            - **2-opt**: Reverse a segment of the tour to eliminate crossings
            - **Insert**: Move a customer to another position in the tour
            - **Relocate**: Move a customer to the beginning of the tour
            
            ### Implementation Details:
            
            - Uses a simplified GNN architecture for feature extraction
            - Implements DQN (Deep Q-Network) for the reinforcement learning component
            - Experience replay buffer for stable learning
            - Epsilon-greedy exploration strategy
            
            This implementation is a simplified version of the GENIS approach for educational purposes.
            """)

if __name__ == "__main__":
    main()