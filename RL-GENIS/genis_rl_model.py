import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
import copy # Needed for target network deepcopy

# --- Constants ---
TARGET_UPDATE = 10 # How often to update target network (in steps or episodes)
INPUT_DIM = 3      # Matching the environment's feature dimension
NUM_ACTIONS = 4    # Matching the environment's action space
HIDDEN_DIM = 16
OUTPUT_DIM = 32
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01 # Lowered end epsilon for more exploitation later
EPSILON_DECAY = 0.995
REPLAY_BUFFER_CAPACITY = 1000 # Increased capacity, closer to paper
BATCH_SIZE = 32               # Increased batch size, closer to paper


class SimpleGNN(nn.Module):
    """
    A simplified Graph Neural Network for feature extraction
    """
    # Corrected input_dim default
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
        super(SimpleGNN, self).__init__()

        # GNN layers
        self.gnn1 = nn.Linear(input_dim, hidden_dim)
        self.gnn2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, node_features, adjacency_matrix):
        """
        Forward pass through the GNN

        Args:
            node_features: Tensor of shape [num_nodes, input_dim]
            adjacency_matrix: Tensor of shape [num_nodes, num_nodes] (normalized recommended)

        Returns:
            Tensor of shape [output_dim] - graph embedding
        """
        # Basic GCN requires normalized adjacency matrix for stability
        # A_hat = A + I
        # D_hat = diagonal matrix of A_hat row sums
        # D_hat_inv_sqrt = D_hat^(-0.5)
        # norm_adj = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
        # For simplicity, we'll skip normalization here, but it's important in practice.

        # First GNN layer + Activation
        h = F.relu(self.gnn1(node_features))

        # Simplistic Message passing with adjacency matrix
        # Note: Proper GCN usually does D^-0.5 * A * D^-0.5 * H * W
        # This is a much simpler version: A * H
        h = torch.matmul(adjacency_matrix.float(), h) # Ensure adj is float

        # Second GNN layer + Activation
        h = F.relu(self.gnn2(h))

        # Global pooling (mean) to get graph embedding
        graph_embedding = torch.mean(h, dim=0)

        return graph_embedding


class DQN(nn.Module):
    """
    Deep Q-Network that uses a GNN for feature extraction
    """
    # Corrected defaults
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_actions=NUM_ACTIONS):
        super(DQN, self).__init__()

        # GNN for feature extraction
        self.gnn = SimpleGNN(input_dim, hidden_dim, output_dim)

        # Q-value prediction layer
        self.q_layer = nn.Linear(output_dim, num_actions)

    def forward(self, node_features, adjacency_matrix):
        """
        Forward pass through the DQN

        Args:
            node_features: Tensor of shape [num_nodes, input_dim]
            adjacency_matrix: Tensor of shape [num_nodes, num_nodes]

        Returns:
            Tensor of shape [num_actions] - Q-values for each action
        """
        # Get graph embedding from GNN
        graph_embedding = self.gnn(node_features, adjacency_matrix)

        # Predict Q-values
        q_values = self.q_layer(graph_embedding)

        return q_values


class ReplayBuffer:
    """Simple experience replay buffer for storing transitions"""

    def __init__(self, capacity=REPLAY_BUFFER_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer"""
        # Store state components individually if they are large/complex
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        actual_batch_size = min(len(self.buffer), batch_size)
        return random.sample(self.buffer, actual_batch_size)

    def __len__(self):
        return len(self.buffer)


class SimpleGENISAgent:
    """
    Simplified GENIS RL Agent for selecting search operators
    """

    def __init__(self, num_actions=NUM_ACTIONS, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
                 learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, epsilon_decay=EPSILON_DECAY):
        """
        Initialize the agent

        Args:
            num_actions: Number of search operators
            input_dim: Dimension of node features
            hidden_dim: Hidden dimension for GNN
            output_dim: Output dimension for graph embedding
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Starting exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of decay for exploration
        """
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.total_steps = 0 # To track when to update target network

        # Initialize Q-network and Target Q-network
        self.q_network = DQN(input_dim, hidden_dim, output_dim, num_actions)
        self.target_q_network = DQN(input_dim, hidden_dim, output_dim, num_actions)
        # Copy weights from Q-network to Target Q-network initially
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval() # Target network is not trained directly

        # Initialize optimizer (only for the main Q-network)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY)

    def select_action(self, state):
        """
        Select an action based on current state using epsilon-greedy policy

        Args:
            state: Tuple of (node_features, adjacency_matrix, cost)

        Returns:
            Selected action index
        """
        node_features, adjacency_matrix, _ = state

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Random action
            return random.randint(0, self.num_actions - 1)
        else:
            # Greedy action
            self.q_network.eval() # Set network to evaluation mode
            with torch.no_grad():
                # Ensure inputs are tensors
                if not isinstance(node_features, torch.Tensor):
                    node_features = torch.tensor(node_features, dtype=torch.float32)
                if not isinstance(adjacency_matrix, torch.Tensor):
                    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)

                q_values = self.q_network(node_features, adjacency_matrix)
            self.q_network.train() # Set network back to training mode
            return torch.argmax(q_values).item()

    def update_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        # It might be better to store tensors directly if possible
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self, batch_size=BATCH_SIZE):
        """Update Q-network from experiences in replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return None # Not enough samples yet

        # Sample from replay buffer
        batch = self.replay_buffer.sample(batch_size)

        # --- Prepare batch tensors ---
        # This part is slow because it processes states individually.
        # A better approach uses torch_geometric.data.Batch for graph batching.
        states_node_features_batch = []
        states_adj_matrices_batch = []
        next_states_node_features_batch = []
        next_states_adj_matrices_batch = []
        actions_batch = []
        rewards_batch = []
        dones_batch = []

        for state, action, reward, next_state, done in batch:
            node_features, adj_matrix, _ = state
            next_node_features, next_adj_matrix, _ = next_state

            # Ensure they are tensors before appending
            states_node_features_batch.append(node_features if isinstance(node_features, torch.Tensor) else torch.tensor(node_features, dtype=torch.float32))
            states_adj_matrices_batch.append(adj_matrix if isinstance(adj_matrix, torch.Tensor) else torch.tensor(adj_matrix, dtype=torch.float32))
            next_states_node_features_batch.append(next_node_features if isinstance(next_node_features, torch.Tensor) else torch.tensor(next_node_features, dtype=torch.float32))
            next_states_adj_matrices_batch.append(next_adj_matrix if isinstance(next_adj_matrix, torch.Tensor) else torch.tensor(next_adj_matrix, dtype=torch.float32))
            actions_batch.append(action)
            rewards_batch.append(reward)
            dones_batch.append(done)

        # --- Compute Q-values for current states ---
        # Apply q_network to each state in the batch (inefficiently)
        current_q_values_list = [self.q_network(nf, adj) for nf, adj in zip(states_node_features_batch, states_adj_matrices_batch)]
        current_q_values_batch = torch.stack(current_q_values_list)

        # Get Q-values for the actions that were actually taken
        actions_tensor = torch.tensor(actions_batch, dtype=torch.long).unsqueeze(1)
        current_q_values = current_q_values_batch.gather(1, actions_tensor).squeeze(1)

        # --- Compute target Q-values using the target network ---
        with torch.no_grad():
            # Apply target_q_network to each next_state in the batch (inefficiently)
            next_q_values_list = [self.target_q_network(nf, adj) for nf, adj in zip(next_states_node_features_batch, next_states_adj_matrices_batch)]
            next_q_values_batch = torch.stack(next_q_values_list)

            # Get the max Q-value for the next state (Double DQN improvement could be added here)
            max_next_q_values = next_q_values_batch.max(1)[0]

            # Convert rewards and dones to tensors
            rewards_tensor = torch.tensor(rewards_batch, dtype=torch.float32)
            # Ensure dones are float (0.0 or 1.0)
            dones_tensor = torch.tensor(dones_batch, dtype=torch.float)

            # Calculate target Q-values: R + gamma * max_Q_target(S') * (1 - done)
            target_q_values = rewards_tensor + self.gamma * max_next_q_values * (1.0 - dones_tensor)

        # --- Compute loss ---
        loss = F.mse_loss(current_q_values, target_q_values)

        # --- Update Q-network ---
        self.optimizer.zero_grad()
        loss.backward()
        # Optional: Clip gradients
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.total_steps += 1
        # --- Update Target Network ---
        # Periodically copy weights from q_network to target_q_network
        if self.total_steps % TARGET_UPDATE == 0:
             self.target_q_network.load_state_dict(self.q_network.state_dict())
             # print("Updated target network") # For debugging

        return loss.item()


class SimpleVRPEnvironment:
    """
    Simplified environment for the Vehicle Routing Problem
    """

    def __init__(self, num_customers=10, grid_size=100):
        self.num_customers = num_customers
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        """Reset the environment"""
        # Generate random coordinates for depot (0) and customers (1-num_customers)
        self.coords = torch.rand(self.num_customers + 1, 2) * self.grid_size

        # Initialize solution as a tour: depot -> all customers -> depot
        self.tour = [0]  # Start at depot
        # Ensure customers are 1 to num_customers
        customers = list(range(1, self.num_customers + 1))
        random.shuffle(customers)
        self.tour.extend(customers)
        self.tour.append(0)  # Return to depot

        # Calculate solution cost
        self.current_cost = self._calculate_cost()

        # Create initial state
        state = self._get_state()

        return state

    def _calculate_cost(self):
        """Calculate the cost of the current solution (total distance)"""
        cost = 0.0
        for i in range(len(self.tour) - 1):
            node1 = self.tour[i]
            node2 = self.tour[i + 1]
            # Ensure coords are tensors if not already
            if not isinstance(self.coords, torch.Tensor):
                self.coords = torch.tensor(self.coords, dtype=torch.float32)
            cost += torch.norm(self.coords[node1] - self.coords[node2]).item()
        return cost

    def _get_state(self):
        """Get the current state representation"""
        num_nodes = self.num_customers + 1
        # Node features: [x, y, is_depot] (INPUT_DIM = 3)
        node_features = torch.zeros(num_nodes, INPUT_DIM, dtype=torch.float32)
        # Ensure coords are tensor
        if not isinstance(self.coords, torch.Tensor):
            self.coords = torch.tensor(self.coords, dtype=torch.float32)
        node_features[:, 0:2] = self.coords  # x, y coordinates
        node_features[0, 2] = 1.0  # Mark depot

        # Adjacency matrix from current solution
        adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
        for i in range(len(self.tour) - 1):
            node1 = self.tour[i]
            node2 = self.tour[i + 1]
            # Check bounds just in case tour modification is buggy
            if 0 <= node1 < num_nodes and 0 <= node2 < num_nodes:
                adjacency_matrix[node1, node2] = 1.0
                adjacency_matrix[node2, node1] = 1.0  # Undirected graph
            else:
                print(f"Warning: Invalid node index encountered in tour: {node1}, {node2}")

        return node_features, adjacency_matrix, self.current_cost

    def step(self, action):
        """
        Take a step in the environment by applying an action

        Args:
            action: Index of the search operator to apply (0 to NUM_ACTIONS-1)

        Returns:
            next_state, reward, done
        """
        # Save old cost for reward calculation
        old_cost = self.current_cost
        original_tour = self.tour[:] # Make a copy in case the move is invalid/trivial

        # Apply different operators based on action
        num_tour_nodes = len(self.tour) # Includes start/end depot
        num_customers_in_tour = num_tour_nodes - 2 # Exclude start/end depot

        if num_customers_in_tour < 2: # Need at least 2 customers for most ops
             pass # Skip action if tour is too small
        elif action == 0:  # Swap two random customers
            idx1, idx2 = random.sample(range(1, num_tour_nodes - 1), 2) # Indices within customer part
            self.tour[idx1], self.tour[idx2] = self.tour[idx2], self.tour[idx1]

        elif action == 1:  # 2-opt: reverse a segment
            # Ensure indices allow a segment of at least 2 nodes to reverse
            if num_customers_in_tour >= 2:
                idx1 = random.randint(1, num_tour_nodes - 2) # Start index (customer)
                idx2 = random.randint(idx1 , num_tour_nodes - 2) # End index (customer)
                if idx1 < idx2 : # Need at least 2 points to reverse segment
                     self.tour[idx1:idx2+1] = self.tour[idx1:idx2+1][::-1] # Use slicing reverse

        elif action == 2:  # Insert: move a customer to another position
             if num_customers_in_tour >= 2:
                from_idx = random.randint(1, num_tour_nodes - 2)
                to_idx = random.randint(1, num_tour_nodes - 2)
                if from_idx != to_idx:
                    customer = self.tour.pop(from_idx)
                    # Adjust to_idx if pop happened before it
                    insert_pos = to_idx if from_idx > to_idx else to_idx -1
                    # Ensure insert_pos is valid after pop
                    insert_pos = max(1, min(insert_pos, len(self.tour)-1))
                    self.tour.insert(insert_pos, customer)


        elif action == 3:  # Relocate: move a customer towards the beginning (after depot)
             if num_customers_in_tour >= 2:
                from_idx = random.randint(1, num_tour_nodes - 2) # Pick a customer
                customer = self.tour.pop(from_idx)
                self.tour.insert(1, customer) # Insert after depot

        # Check if tour is still valid (basic check)
        if len(self.tour) != num_tour_nodes or self.tour[0] != 0 or self.tour[-1] != 0:
             print(f"Warning: Tour modification resulted in invalid tour: {self.tour}. Reverting.")
             self.tour = original_tour


        # Calculate new cost
        self.current_cost = self._calculate_cost()

        # Calculate reward based on improvement
        # Use a small epsilon to avoid division by zero or huge rewards if old_cost is tiny
        epsilon = 1e-6
        reward = (old_cost - self.current_cost) / (old_cost + epsilon)

        # Get new state
        next_state = self._get_state()

        # Check if done (we're using a fixed number of steps, so always False)
        done = False

        return next_state, reward, done


def train_simple_genis(num_episodes=100, max_steps=200): # Increased episodes/steps
    """
    Train the simplified GENIS agent

    Args:
        num_episodes: Number of episodes to train
        max_steps: Maximum steps per episode
    """
    env = SimpleVRPEnvironment(num_customers=20) # Slightly larger problem
    agent = SimpleGENISAgent(num_actions=NUM_ACTIONS) # Use constant

    best_cost = float('inf')
    costs_over_time = []
    total_steps_counter = 0

    print(f"Starting training for {num_episodes} episodes...")
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward_sum = 0.0

        for step in range(max_steps):
            # Select action
            action = agent.select_action(state)

            # Take step in environment
            next_state, reward, done = env.step(action)

            # Store transition
            # Ensure state and next_state are stored correctly (e.g., deep copies if needed)
            agent.store_transition(state, action, reward, copy.deepcopy(next_state), done)

            # Update agent
            loss = agent.update(batch_size=BATCH_SIZE) # Use constant

            state = next_state
            episode_reward_sum += reward
            total_steps_counter += 1

            if done: # Although 'done' is always False here
                break

        # Update exploration rate after each episode
        agent.update_epsilon()

        # Track best solution cost found so far
        final_cost = state[2] # Cost is the 3rd element of the state tuple
        if final_cost < best_cost:
            best_cost = final_cost
            # print(f"  New best cost found: {best_cost:.2f}") # Debugging

        costs_over_time.append(final_cost)

        # Print episode statistics
        if (episode + 1) % 10 == 0:
            avg_reward = episode_reward_sum / max_steps if max_steps > 0 else 0
            print(f"Episode {episode+1}/{num_episodes}, Steps: {step+1}, Final Cost: {final_cost:.2f}, "
                  f"Avg Reward: {avg_reward:.4f}, Best Cost: {best_cost:.2f}, Epsilon: {agent.epsilon:.3f}")
            if loss is not None:
                 print(f"  Last Batch Loss: {loss:.4f}")

    print(f"\nTraining complete! Best cost found: {best_cost:.2f}")
    return agent, env, costs_over_time


def visualize_solution(env):
    """Visualize the current solution"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not found. Cannot visualize solution. Install using: pip install matplotlib")
        return

    plt.figure(figsize=(10, 8)) # Slightly larger figure

    coords_np = env.coords.numpy() if isinstance(env.coords, torch.Tensor) else env.coords
    tour = env.tour

    # Plot depot
    plt.scatter(coords_np[0, 0], coords_np[0, 1], c='red', s=200, marker='s', label='Depot (0)', zorder=3)

    # Plot customers
    if len(coords_np) > 1:
        plt.scatter(coords_np[1:, 0], coords_np[1:, 1], c='blue', s=50, label='Customers', zorder=2)

    # Plot the tour lines
    if len(tour) > 1:
         tour_coords = coords_np[tour]
         plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'k-', linewidth=1.0, alpha=0.7, zorder=1)

    # Annotate customers (nodes 1 to N)
    for i in range(1, len(coords_np)):
        plt.annotate(f"{i}", (coords_np[i, 0], coords_np[i, 1]),
                     xytext=(5, 5), textcoords='offset points')

    plt.title(f"VRP Solution - Cost: {env.current_cost:.2f}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Ensure aspect ratio is equal
    plt.savefig("vrp_solution.png")
    print("Saved solution visualization to vrp_solution.png")
    # plt.show() # Uncomment to display interactively


def plot_learning_curve(costs):
    """Plot the learning curve showing cost reduction over episodes"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not found. Cannot plot learning curve. Install using: pip install matplotlib")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(costs, 'b-')
    plt.title('Learning Curve (Solution Cost per Episode)')
    plt.xlabel('Episode')
    plt.ylabel('Solution Cost')
    plt.grid(True)
    plt.savefig("learning_curve.png")
    print("Saved learning curve to learning_curve.png")
    # plt.show() # Uncomment to display interactively


if __name__ == "__main__":
    print("Running Simple GENIS RL Model Training...")
    # Train the agent
    # Increased episodes and steps for potentially better learning
    trained_agent, final_env, cost_history = train_simple_genis(num_episodes=200, max_steps=250)

    # Visualize final solution from the environment state after training
    print("\nVisualizing final solution...")
    visualize_solution(final_env)

    # Plot learning curve
    print("Plotting learning curve...")
    plot_learning_curve(cost_history)

    print("\nScript finished.")