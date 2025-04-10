import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import math

class GraphNeighborhoodSearch:
    def __init__(self, n_nodes=20, k=5):
        """
        Initialize the Graph Neighborhood Search.
        
        Args:
            n_nodes: Number of nodes (including depot)
            k: Number of nearest neighbors to consider
        """
        self.n_nodes = n_nodes
        self.k = k
        self.depot_idx = 0  # Depot is at index 0
        self.coords = None
        self.demands = None
        self.distance_matrix = None
        self.nearest_neighbors = None
        self.vehicle_capacity = 1.0
        self.routes = []
        self.total_distance = 0
        
    def generate_random_problem(self):
        """Generate a random VRP instance"""
        # Generate random coordinates in [0,1] x [0,1]
        self.coords = np.random.rand(self.n_nodes, 2)
        
        # Make depot at (0.5, 0.5) - center
        self.coords[0] = np.array([0.5, 0.5])
        
        # Generate random demands in [0.1, 0.4]
        self.demands = np.random.uniform(0.1, 0.4, size=self.n_nodes)
        self.demands[0] = 0  # Depot has no demand
        
        # Calculate distance matrix
        self.distance_matrix = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j:
                    # Euclidean distance
                    self.distance_matrix[i, j] = np.sqrt(
                        ((self.coords[i, 0] - self.coords[j, 0]) ** 2) + 
                        ((self.coords[i, 1] - self.coords[j, 1]) ** 2)
                    )
        
        # Find k nearest neighbors for each node
        self.find_nearest_neighbors()
        
    def find_nearest_neighbors(self):
        """Find k nearest neighbors for each node"""
        self.nearest_neighbors = []
        for i in range(self.n_nodes):
            # Get distances from node i to all other nodes
            distances = self.distance_matrix[i]
            # Find indices of k nearest neighbors (excluding self)
            sorted_indices = np.argsort(distances)
            # Skip the first element (which is the node itself with distance 0)
            neighbors = sorted_indices[1:self.k+1]
            self.nearest_neighbors.append(neighbors)
    
    def greedy_vrp_solve(self):
        """
        Solve VRP with a simple greedy algorithm using neighborhood information
        """
        # Reset routes
        self.routes = []
        self.total_distance = 0
        
        # Initialize
        remaining_nodes = set(range(1, self.n_nodes))  # Skip depot
        current_route = [0]  # Start from depot
        current_capacity = self.vehicle_capacity
        
        while remaining_nodes:
            last_node = current_route[-1]
            
            # Find nearest unvisited neighbor that doesn't exceed capacity
            found_next = False
            
            # First check among nearest neighbors
            for neighbor in self.nearest_neighbors[last_node]:
                if neighbor in remaining_nodes and self.demands[neighbor] <= current_capacity:
                    next_node = neighbor
                    found_next = True
                    break
            
            # If no suitable neighbor found, check all remaining nodes
            if not found_next:
                min_distance = float('inf')
                next_node = None
                
                for node in remaining_nodes:
                    if self.demands[node] <= current_capacity:
                        dist = self.distance_matrix[last_node, node]
                        if dist < min_distance:
                            min_distance = dist
                            next_node = node
                
                if next_node is not None:
                    found_next = True
            
            # If we found a next node, update route
            if found_next:
                current_route.append(next_node)
                current_capacity -= self.demands[next_node]
                remaining_nodes.remove(next_node)
            else:
                # Return to depot and start a new route
                current_route.append(0)
                self.routes.append(current_route)
                
                # Calculate distance of this route
                route_distance = sum(self.distance_matrix[current_route[i], current_route[i+1]] 
                                    for i in range(len(current_route)-1))
                self.total_distance += route_distance
                
                # Start new route
                if remaining_nodes:
                    current_route = [0]
                    current_capacity = self.vehicle_capacity
        
        # Add final route if not empty
        if len(current_route) > 1:
            current_route.append(0)  # Return to depot
            self.routes.append(current_route)
            
            # Calculate distance of this route
            route_distance = sum(self.distance_matrix[current_route[i], current_route[i+1]] 
                                for i in range(len(current_route)-1))
            self.total_distance += route_distance
    
    def two_opt_local_search(self, route):
        """
        Apply 2-opt local search to improve a single route
        
        Args:
            route: List of node indices including depot at start and end
            
        Returns:
            Improved route
        """
        improved = True
        best_route = route.copy()
        best_distance = self.calculate_route_distance(best_route)
        
        while improved:
            improved = False
            
            # Try all possible 2-opt swaps (excluding depot)
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    new_route = best_route.copy()
                    # Reverse the segment between i and j
                    new_route[i:j+1] = reversed(new_route[i:j+1])
                    
                    new_distance = self.calculate_route_distance(new_route)
                    
                    if new_distance < best_distance:
                        best_distance = new_distance
                        best_route = new_route
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_route, best_distance
    
    def calculate_route_distance(self, route):
        """Calculate the total distance of a route"""
        return sum(self.distance_matrix[route[i], route[i+1]] for i in range(len(route)-1))
    
    def improve_routes(self):
        """Apply local search to improve all routes"""
        improved_routes = []
        new_total_distance = 0
        
        for route in self.routes:
            improved_route, route_distance = self.two_opt_local_search(route)
            improved_routes.append(improved_route)
            new_total_distance += route_distance
        
        self.routes = improved_routes
        self.total_distance = new_total_distance
    
    def plot_solution(self, fig=None, ax=None):
        """Plot the VRP solution"""
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot nodes
        ax.scatter(self.coords[1:, 0], self.coords[1:, 1], c='blue', s=50, label='Customers')
        ax.scatter(self.coords[0, 0], self.coords[0, 1], c='red', s=100, marker='*', label='Depot')
        
        # Add demand labels
        for i in range(1, self.n_nodes):
            ax.text(self.coords[i, 0] + 0.01, self.coords[i, 1] + 0.01, 
                    f"{i}\n({self.demands[i]:.1f})", fontsize=9)
        
        # Plot routes with different colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.routes)))
        
        for i, route in enumerate(self.routes):
            route_x = self.coords[route, 0]
            route_y = self.coords[route, 1]
            ax.plot(route_x, route_y, c=colors[i], linewidth=2, 
                    label=f"Route {i+1}")
        
        # Plot neighborhood connections (lightly)
        for i in range(self.n_nodes):
            for j in self.nearest_neighbors[i]:
                ax.plot([self.coords[i, 0], self.coords[j, 0]], 
                        [self.coords[i, 1], self.coords[j, 1]], 
                        'gray', alpha=0.1, linestyle='--')
        
        ax.set_title(f"VRP Solution: {len(self.routes)} routes, Total Distance: {self.total_distance:.3f}")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig, ax