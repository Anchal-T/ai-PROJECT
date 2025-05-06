import streamlit as st
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

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

# Set page config
st.set_page_config(
    page_title="GENIS RL for VRP",
    page_icon="🚚",
    layout="wide"
)

# Set random seeds for reproducibility
@st.cache_resource
def set_seeds(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return True

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

# Main function
def main():
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

# Run the app
if __name__ == "__main__":
    main()