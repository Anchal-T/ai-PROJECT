"""
Simple script to run the GENIS model
"""

from genis_rl_model import train_simple_genis, visualize_solution, plot_learning_curve
import torch
import random
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

def main():
    print("Training Simple GENIS RL model...")
    
    # Train the model (feel free to adjust these parameters)
    num_episodes = 30  # Number of training episodes
    max_steps = 100    # Maximum steps per episode
    
    agent, env, costs = train_simple_genis(num_episodes=num_episodes, max_steps=max_steps)
    
    print("\nTraining complete!")
    print(f"Final solution cost: {env.current_cost:.2f}")
    
    # Visualize the final solution
    try:
        visualize_solution(env)
        print("Solution visualization saved as 'vrp_solution.png'")
        
        # Plot learning curve
        plot_learning_curve(costs)
        print("Learning curve saved as 'learning_curve.png'")
    except ImportError:
        print("Matplotlib not installed. Skipping visualization.")

if __name__ == "__main__":
    main()