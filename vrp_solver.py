import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from typing import List, Tuple, Dict
import pandas as pd

class VehicleRoutingProblem:
    def __init__(self, num_customers=20, capacity=30):
        self.num_customers = num_customers
        self.capacity = capacity
        self.depot_coords = (50, 50)  # Central depot
        self.customer_coords = []
        self.customer_demands = []
        self.distance_matrix = None
        self.best_route = None
        self.best_distance = float('inf')
        
    def generate_random_problem(self, seed=None):
        """Generate a random VRP instance"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Generate customer coordinates (x, y) between 0 and 100
        self.customer_coords = [(random.uniform(0, 100), random.uniform(0, 100)) 
                               for _ in range(self.num_customers)]
        
        # Generate customer demands between 1 and 10
        self.customer_demands = [random.randint(1, 10) for _ in range(self.num_customers)]
        
        # Calculate distance matrix
        self._calculate_distance_matrix()
        
    def _calculate_distance_matrix(self):
        """Calculate distance matrix between all nodes (depot + customers)"""
        all_coords = [self.depot_coords] + self.customer_coords
        n = len(all_coords)
        self.distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = all_coords[i]
                    x2, y2 = all_coords[j]
                    self.distance_matrix[i, j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def nearest_neighbor_search(self, progress_callback=None):
        """
        Simple nearest neighbor search algorithm for VRP
        Returns routes for each vehicle
        """
        routes = []
        unvisited = list(range(1, self.num_customers + 1))  # Customer indices (1-indexed)
        total_distance = 0
        
        while unvisited:
            # Start a new vehicle route
            route = [0]  # Start at depot
            remaining_capacity = self.capacity
            
            while unvisited:
                last_node = route[-1]
                # Find nearest unvisited customer that fits into the remaining capacity
                nearest_idx = None
                nearest_dist = float('inf')
                
                for i, customer_idx in enumerate(unvisited):
                    # Check if customer's demand fits into remaining capacity
                    demand = self.customer_demands[customer_idx - 1]  # -1 because customers are 1-indexed
                    if demand <= remaining_capacity:
                        dist = self.distance_matrix[last_node, customer_idx]
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_idx = i
                
                # If no customer fits or no more customers, end this route
                if nearest_idx is None:
                    break
                    
                # Add customer to route
                customer_idx = unvisited.pop(nearest_idx)
                route.append(customer_idx)
                total_distance += nearest_dist
                remaining_capacity -= self.customer_demands[customer_idx - 1]
                
                # Update visualization if callback provided
                if progress_callback:
                    progress_callback(routes + [route], total_distance)
                    time.sleep(0.2)  # Add delay for visualization
            
            # Return to depot
            total_distance += self.distance_matrix[route[-1], 0]
            route.append(0)  # End at depot
            routes.append(route)
            
            # Update visualization with completed route
            if progress_callback:
                progress_callback(routes, total_distance)
                time.sleep(0.2)
        
        self.best_route = routes
        self.best_distance = total_distance
        return routes, total_distance
    
    def savings_algorithm(self, progress_callback=None):
        """
        Clarke-Wright savings algorithm for VRP
        Returns routes for each vehicle
        """
        # Start with each customer served by a dedicated vehicle
        routes = [[0, i, 0] for i in range(1, self.num_customers + 1)]
        total_distance = sum(self.distance_matrix[0, i] + self.distance_matrix[i, 0] 
                            for i in range(1, self.num_customers + 1))
        
        # Calculate savings for each pair of customers
        savings = []
        for i in range(1, self.num_customers + 1):
            for j in range(i+1, self.num_customers + 1):
                saving = self.distance_matrix[i, 0] + self.distance_matrix[0, j] - self.distance_matrix[i, j]
                savings.append((saving, i, j))
        
        # Sort savings in descending order
        savings.sort(reverse=True)
        
        # Merge routes based on savings
        for saving, i, j in savings:
            # Find routes containing i and j
            route_i = route_j = None
            route_i_idx = route_j_idx = None
            
            for idx, route in enumerate(routes):
                if i in route:
                    route_i = route
                    route_i_idx = idx
                if j in route:
                    route_j = route
                    route_j_idx = idx
            
            # Check if i and j are in different routes and at the end/beginning
            if route_i != route_j:
                # Four cases to consider for merging
                merged = False
                
                # Case 1: i is last, j is first
                if route_i[-2] == i and route_j[1] == j:
                    # Check capacity constraint
                    route_capacity = sum(self.customer_demands[node-1] for node in route_i[1:-1] + route_j[1:-1])
                    if route_capacity <= self.capacity:
                        new_route = route_i[:-1] + route_j[1:]
                        merged = True
                
                # Case 2: j is last, i is first
                elif route_j[-2] == j and route_i[1] == i:
                    # Check capacity constraint
                    route_capacity = sum(self.customer_demands[node-1] for node in route_j[1:-1] + route_i[1:-1])
                    if route_capacity <= self.capacity:
                        new_route = route_j[:-1] + route_i[1:]
                        merged = True
                
                # Case 3: i and j are both last
                elif route_i[-2] == i and route_j[-2] == j:
                    # Check capacity constraint
                    route_capacity = sum(self.customer_demands[node-1] for node in route_i[1:-1] + route_j[1:-1][::-1])
                    if route_capacity <= self.capacity:
                        new_route = route_i[:-1] + route_j[-2:0:-1] + [0]
                        merged = True
                
                # Case 4: i and j are both first
                elif route_i[1] == i and route_j[1] == j:
                    # Check capacity constraint
                    route_capacity = sum(self.customer_demands[node-1] for node in route_i[1:-1][::-1] + route_j[1:-1])
                    if route_capacity <= self.capacity:
                        new_route = [0] + route_i[-2:0:-1] + route_j[1:]
                        merged = True
                
                # If merged, update routes and distance
                if merged:
                    # Calculate new total distance
                    new_distance = 0
                    for r in routes:
                        if r != route_i and r != route_j:
                            for k in range(len(r) - 1):
                                new_distance += self.distance_matrix[r[k], r[k+1]]
                    
                    # Add distance of new route
                    for k in range(len(new_route) - 1):
                        new_distance += self.distance_matrix[new_route[k], new_route[k+1]]
                    
                    # Update routes
                    routes = [r for r in routes if r != route_i and r != route_j]
                    routes.append(new_route)
                    total_distance = new_distance
                    
                    # Update visualization if callback provided
                    if progress_callback:
                        progress_callback(routes, total_distance)
                        time.sleep(0.2)
        
        self.best_route = routes
        self.best_distance = total_distance
        return routes, total_distance
    
    def two_opt_improvement(self, routes, progress_callback=None):
        """Apply 2-opt improvement to each route"""
        improved = True
        total_distance = 0
        
        while improved:
            improved = False
            for r_idx, route in enumerate(routes):
                if len(route) <= 4:  # Need at least 2 customers to apply 2-opt
                    continue
                
                # Try all 2-opt swaps within this route
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route) - 1):
                        # Calculate current distance
                        current_dist = (self.distance_matrix[route[i-1], route[i]] + 
                                       self.distance_matrix[route[j], route[j+1]])
                        
                        # Calculate new distance if swapped
                        new_dist = (self.distance_matrix[route[i-1], route[j]] + 
                                   self.distance_matrix[route[i], route[j+1]])
                        
                        # If improvement found
                        if new_dist < current_dist:
                            # Apply 2-opt swap
                            route[i:j+1] = reversed(route[i:j+1])
                            improved = True
                            
                            # Update visualization if callback provided
                            if progress_callback:
                                current_total = sum(self.calculate_route_distance(r) for r in routes)
                                progress_callback(routes, current_total)
                                time.sleep(0.2)
                            
                            # Start over with this route
                            break
                    
                    if improved:
                        break
        
        # Calculate final distance
        total_distance = sum(self.calculate_route_distance(route) for route in routes)
        
        if total_distance < self.best_distance:
            self.best_route = routes
            self.best_distance = total_distance
        
        return routes, total_distance
    
    def calculate_route_distance(self, route):
        """Calculate the total distance of a route"""
        distance = 0
        for i in range(len(route) - 1):
            distance += self.distance_matrix[route[i], route[i+1]]
        return distance

    def plot_routes(self, routes=None, ax=None, show_demands=True):
        """Plot the VRP routes"""
        if routes is None:
            routes = self.best_route
            
        # Create a new figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot depot
        ax.plot(self.depot_coords[0], self.depot_coords[1], 'ks', markersize=10, label='Depot')
        
        # Plot customers
        for i, (x, y) in enumerate(self.customer_coords):
            demand = self.customer_demands[i]
            ax.plot(x, y, 'bo', markersize=max(5, demand * 0.8))
            if show_demands:
                ax.annotate(f"{i+1}:{demand}", (x, y), xytext=(5, 5), textcoords='offset points')
        
        # Plot routes with different colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
        
        for i, route in enumerate(routes):
            route_x = [self.depot_coords[0] if node == 0 else self.customer_coords[node - 1][0] for node in route]
            route_y = [self.depot_coords[1] if node == 0 else self.customer_coords[node - 1][1] for node in route]
            ax.plot(route_x, route_y, '-', color=colors[i], linewidth=2, alpha=0.8)
            
            # Add arrows to show direction
            for j in range(len(route) - 1):
                x1, y1 = route_x[j], route_y[j]
                x2, y2 = route_x[j+1], route_y[j+1]
                dx, dy = x2 - x1, y2 - y1
                ax.arrow(x1, y1, dx * 0.8, dy * 0.8, head_width=2, head_length=2, 
                        fc=colors[i], ec=colors[i], alpha=0.8)
        
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_title('Vehicle Routing Problem Solution')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True)
        
        return ax

def streamlit_vrp_demo():
    st.title("Vehicle Routing Problem Solver")
    
    st.sidebar.header("Problem Settings")
    num_customers = st.sidebar.slider("Number of Customers", 5, 50, 20)
    vehicle_capacity = st.sidebar.slider("Vehicle Capacity", 10, 100, 30)
    random_seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)
    
    # Create VRP instance
    vrp = VehicleRoutingProblem(num_customers=num_customers, capacity=vehicle_capacity)
    vrp.generate_random_problem(seed=random_seed)
    
    # Display problem data
    st.subheader("Problem Instance")
    
    # Create DataFrame for customer data
    customer_data = {
        "Customer": list(range(1, num_customers + 1)),
        "X": [coord[0] for coord in vrp.customer_coords],
        "Y": [coord[1] for coord in vrp.customer_coords],
        "Demand": vrp.customer_demands
    }
    df_customers = pd.DataFrame(customer_data)
    
    # Add depot information
    depot_df = pd.DataFrame({
        "Customer": ["Depot"],
        "X": [vrp.depot_coords[0]],
        "Y": [vrp.depot_coords[1]],
        "Demand": [0]
    })
    
    # Combine and display
    df_display = pd.concat([depot_df, df_customers], ignore_index=True)
    st.dataframe(df_display, use_container_width=True)
    
    # Create columns for visualization and metrics
    col1, col2 = st.columns([3, 1])
    
    # Setup plots area in column 1
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(vrp.depot_coords[0], vrp.depot_coords[1], 'ks', markersize=10, label='Depot')
    for i, (x, y) in enumerate(vrp.customer_coords):
        demand = vrp.customer_demands[i]
        ax.plot(x, y, 'bo', markersize=max(5, demand * 0.8))
        ax.annotate(f"{i+1}:{demand}", (x, y), xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_title('Vehicle Routing Problem')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.grid(True)
    
    plot_placeholder = col1.empty()
    plot_placeholder.pyplot(fig)
    
    # Setup metrics area in column 2
    distance_metric = col2.empty()
    routes_metric = col2.empty()
    iterations_metric = col2.empty()
    
    # Function to update visualization
    def update_visualization(routes, total_distance):
        nonlocal iterations
        iterations += 1
        
        fig, ax = plt.subplots(figsize=(10, 8))
        vrp.plot_routes(routes, ax)
        plot_placeholder.pyplot(fig)
        
        distance_metric.metric("Total Distance", f"{total_distance:.2f}")
        routes_metric.metric("Number of Routes", len(routes))
        iterations_metric.metric("Iterations", iterations)
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Select Algorithm",
        ["Nearest Neighbor", "Clarke-Wright Savings", "Both with 2-opt Improvement"]
    )
    
    # Run button
    if st.sidebar.button("Run Algorithm"):
        iterations = 0
        
        if algorithm == "Nearest Neighbor":
            with st.spinner("Running Nearest Neighbor algorithm..."):
                routes, distance = vrp.nearest_neighbor_search(update_visualization)
                st.success(f"Nearest Neighbor algorithm completed with total distance: {distance:.2f}")
                
        elif algorithm == "Clarke-Wright Savings":
            with st.spinner("Running Clarke-Wright Savings algorithm..."):
                routes, distance = vrp.savings_algorithm(update_visualization)
                st.success(f"Clarke-Wright Savings algorithm completed with total distance: {distance:.2f}")
                
        else:  # Both with 2-opt
            with st.spinner("Running Nearest Neighbor algorithm..."):
                nn_routes, nn_distance = vrp.nearest_neighbor_search(update_visualization)
                st.info(f"Nearest Neighbor completed with distance: {nn_distance:.2f}")
                
            with st.spinner("Applying 2-opt improvement to Nearest Neighbor solution..."):
                nn_improved_routes, nn_improved_distance = vrp.two_opt_improvement(
                    [r.copy() for r in nn_routes], update_visualization
                )
                st.info(f"Nearest Neighbor with 2-opt: {nn_improved_distance:.2f}")
                
            with st.spinner("Running Clarke-Wright Savings algorithm..."):
                cw_routes, cw_distance = vrp.savings_algorithm(update_visualization)
                st.info(f"Clarke-Wright completed with distance: {cw_distance:.2f}")
                
            with st.spinner("Applying 2-opt improvement to Clarke-Wright solution..."):
                cw_improved_routes, cw_improved_distance = vrp.two_opt_improvement(
                    [r.copy() for r in cw_routes], update_visualization
                )
                st.info(f"Clarke-Wright with 2-opt: {cw_improved_distance:.2f}")
                
            # Determine the best solution
            best_distance = min(nn_distance, nn_improved_distance, cw_distance, cw_improved_distance)
            if best_distance == nn_distance:
                best_routes = nn_routes
                best_name = "Nearest Neighbor"
            elif best_distance == nn_improved_distance:
                best_routes = nn_improved_routes
                best_name = "Nearest Neighbor with 2-opt"
            elif best_distance == cw_distance:
                best_routes = cw_routes
                best_name = "Clarke-Wright Savings"
            else:
                best_routes = cw_improved_routes
                best_name = "Clarke-Wright Savings with 2-opt"
                
            st.success(f"Best solution: {best_name} with distance {best_distance:.2f}")
            
            # Display final best solution
            fig, ax = plt.subplots(figsize=(10, 8))
            vrp.plot_routes(best_routes, ax)
            plot_placeholder.pyplot(fig)
            
    # Show solution details
    if vrp.best_route:
        st.subheader("Solution Details")
        for i, route in enumerate(vrp.best_route):
            route_demand = sum(vrp.customer_demands[node-1] for node in route if node != 0)
            route_distance = vrp.calculate_route_distance(route)
            
            # Format route for display
            route_str = " → ".join([str(node) if node != 0 else "Depot" for node in route])
            
            st.write(f"**Route {i+1}:** {route_str}")
            st.write(f"- Total demand: {route_demand}/{vrp.capacity}")
            st.write(f"- Distance: {route_distance:.2f}")
            st.write("---")
        
        st.write(f"**Total distance:** {vrp.best_distance:.2f}")

if __name__ == "__main__":
    streamlit_vrp_demo()