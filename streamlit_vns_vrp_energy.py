# streamlit_vns_vrp_energy.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time
from collections import defaultdict

# --- Constants & Parameters (Based on Paper/Defaults) ---
# These would need to be adjusted for a real scenario
DEFAULT_DRONE_PARAMS = {
    'capacity': 5.0,       # Max payload (e.g., kg, consistent unit with demand) [cite: 82, 94]
    'w0': 6.0,             # Drone curb weight (kg) [cite: 263]
    'gamma': 4.0,          # Energy for takeoff/landing (Wh/kg/km - simplified unit) [cite: 263]
    'rho': 3.5,            # Energy for flight (Wh/kg/km) [cite: 263]
    'H': 0.05,             # Fixed flight height (km) [cite: 266]
    'E': 504.0,            # Max battery energy (Wh) [cite: 264]
    'speed': 40.0          # Average speed (km/h) [Used simplification from cite: 265]
}
# VNS Params
MAX_ITERATIONS_VNS = 50
MAX_ITER_NO_IMPROVEMENT_VNS = 15
# Operator order similar to VNS idea (can be randomized too)
NEIGHBORHOOD_ORDER_VNS = ['relocate', 'exchange', 'two_opt_intra'] # Subset from Table I [cite: 203, 204, 205]
SHAKE_STRENGTH_VNS = 3 # Number of random moves in shake

# Cache to store route calculations
route_cache = {}

# --- Helper Functions ---

def calculate_distance(p1, p2):
    """Calculate Euclidean distance."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_energy_consumption(dist, payload, height, params):
    """Calculates energy based on Eq. 1 from the paper."""
    # R_ij^k = gamma * H_ij^k * (w0 + w_ij^k) + rho * L_ij * (w0 + w_ij^k)
    # Simplified: assume H is constant, payload w_ij^k is average for segment for simplicity
    energy = (params['gamma'] * height + params['rho'] * dist) * (params['w0'] + payload) # [cite: 85]
    return energy

def get_route_key(route_indices):
    """Get a hashable key for caching route details."""
    return tuple(route_indices)

def calculate_route_details(route_indices, cities, depot_coord, demands, params):
    """
    Calculates total distance, arrival times, payload, energy consumption, and feasibility for a single route.
    Returns details dict or None if infeasible.
    Uses caching for performance.
    """
    # Check cache first
    route_key = get_route_key(route_indices)
    if route_key in route_cache:
        return route_cache[route_key]
    
    if not route_indices:
        result = {'distance': 0, 'arrival_times': {}, 'total_time': 0, 'energy': 0, 'feasible': True, 'infeasibility_reason': None}
        route_cache[route_key] = result
        return result

    total_dist = 0
    current_payload = sum(demands[i] for i in route_indices)
    current_energy = params['E']
    current_time = 0
    last_coord = depot_coord
    details = {'arrival_times': {}, 'total_time': 0, 'max_payload_violation': False, 'energy_violation': False}

    # Check initial capacity
    if current_payload > params['capacity']:
        details['max_payload_violation'] = True
        result = {'feasible': False, 'infeasibility_reason': f'Initial payload {current_payload:.2f} exceeds capacity {params["capacity"]:.2f}'}
        route_cache[route_key] = result
        return result

    route_path = [depot_coord] + [cities[i] for i in route_indices] + [depot_coord]

    # Fly to first customer
    dist_to_first = calculate_distance(depot_coord, cities[route_indices[0]])
    energy_needed = calculate_energy_consumption(dist_to_first, current_payload, params['H'], params)
    if energy_needed > current_energy:
        result = {'feasible': False, 'infeasibility_reason': f'Not enough energy for first leg ({energy_needed:.2f} > {current_energy:.2f})'}
        route_cache[route_key] = result
        return result
    
    current_energy -= energy_needed
    total_dist += dist_to_first
    time_taken = dist_to_first / params['speed'] * 3600 # time in seconds
    current_time += time_taken
    details['arrival_times'][route_indices[0]] = current_time
    last_coord = cities[route_indices[0]]

    # Fly between customers
    for i in range(len(route_indices) - 1):
        idx_u, idx_v = route_indices[i], route_indices[i+1]
        # Payload decreases *after* visiting customer i
        current_payload -= demands[idx_u]
        dist = calculate_distance(cities[idx_u], cities[idx_v])
        energy_needed = calculate_energy_consumption(dist, current_payload, params['H'], params)
        if energy_needed > current_energy:
             result = {'feasible': False, 'infeasibility_reason': f'Energy fail at C{idx_u}->C{idx_v} ({energy_needed:.2f} > {current_energy:.2f})'}
             route_cache[route_key] = result
             return result
        
        current_energy -= energy_needed
        total_dist += dist
        time_taken = dist / params['speed'] * 3600
        current_time += time_taken
        details['arrival_times'][idx_v] = current_time
        last_coord = cities[idx_v]

    # Fly back to depot
    idx_last = route_indices[-1]
    current_payload -= demands[idx_last] # Payload after last customer
    dist_to_depot = calculate_distance(cities[idx_last], depot_coord)
    energy_needed = calculate_energy_consumption(dist_to_depot, current_payload, params['H'], params)
    if energy_needed > current_energy:
        result = {'feasible': False, 'infeasibility_reason': f'Energy fail C{idx_last}->Depot ({energy_needed:.2f} > {current_energy:.2f})'}
        route_cache[route_key] = result
        return result
    
    current_energy -= energy_needed
    total_dist += dist_to_depot
    # Don't add time for return to depot to objective as per Eq. 2 (sum arrival times at customers) [cite: 90]

    details['distance'] = total_dist
    details['energy'] = params['E'] - current_energy # Total consumed
    details['total_time'] = current_time # Last arrival time
    details['feasible'] = True
    details['infeasibility_reason'] = None
    
    # Cache the result
    route_cache[route_key] = details
    return details


def calculate_total_objective(solution, cities, depots, demands, params):
    """Calculates the objective function (Eq. 2) and checks feasibility."""
    total_sum_arrival_times = 0
    all_feasible = True
    max_vehicles_per_depot = params.get('max_vehicles_per_depot', 100) # Default high number
    details_list = []

    for depot_idx, depot_routes in enumerate(solution):
        if len(depot_routes) > max_vehicles_per_depot: # Check max vehicles per depot [cite: 95]
            all_feasible = False
            # Add dummy infeasible details
            details_list.append([{'feasible': False, 'infeasibility_reason': f'Depot {depot_idx} exceeds max vehicles {len(depot_routes)} > {max_vehicles_per_depot}'}] * len(depot_routes))
            continue

        depot_details = []
        depot_coord = depots[depot_idx]
        for route_indices in depot_routes:
            route_details = calculate_route_details(route_indices, cities, depot_coord, demands, params)
            if route_details is None or not route_details['feasible']:
                all_feasible = False
                depot_details.append(route_details if route_details else {'feasible': False, 'infeasibility_reason': 'Unknown calculation error'})
            else:
                # Sum arrival times for this route
                route_sum_times = sum(route_details['arrival_times'].values())
                total_sum_arrival_times += route_sum_times
                depot_details.append(route_details)
        details_list.append(depot_details)

    return total_sum_arrival_times if all_feasible else float('inf'), all_feasible, details_list


def plot_vrp_solution(cities, depots, solution, title, ax):
    """Plots the VRP solution with multiple depots and routes."""
    ax.clear()
    num_depots = len(depots)
    colors = plt.cm.get_cmap('tab10', num_depots + len(solution)) # Color per depot/route

    # Plot cities (customers)
    customer_coords = np.array(cities)
    ax.plot(customer_coords[:, 0], customer_coords[:, 1], 'ko', markersize=5, label='Customers')
    for i, city in enumerate(cities):
        ax.text(city[0] + 0.5, city[1] + 0.5, str(i), fontsize=8)

    # Plot depots
    depot_coords = np.array(depots)
    ax.plot(depot_coords[:, 0], depot_coords[:, 1], 'rs', markersize=10, label='Depots')
    for i, depot in enumerate(depots):
        ax.text(depot[0] + 0.5, depot[1] + 0.5, f"D{i}", fontsize=9, color='red')

    # Plot routes with details
    route_counter = 0
    for depot_idx, depot_routes in enumerate(solution):
        depot_coord = depots[depot_idx]
        for route_indices in depot_routes:
            if not route_indices: continue
            route_color = colors(route_counter % colors.N) # Cycle through colors
            # Path: Depot -> C1 -> C2 -> ... -> Cn -> Depot
            path_coords = [depot_coord] + [cities[i] for i in route_indices] + [depot_coord]
            path_coords = np.array(path_coords)
            ax.plot(path_coords[:, 0], path_coords[:, 1], '-', color=route_color, lw=1.5, alpha=0.8, label=f"D{depot_idx}-R{route_counter}")
            
            # Add distance labels to route segments for better visualization
            total_distance = 0
            for i in range(len(path_coords)-1):
                p1, p2 = path_coords[i], path_coords[i+1]
                dist = calculate_distance(p1, p2)
                total_distance += dist
                # Add small distance labels (optional)
                #midpoint = (p1 + p2) / 2
                #ax.text(midpoint[0], midpoint[1], f"{dist:.1f}", fontsize=6, color=route_color)
            
            # Add total route distance near the first segment
            if len(path_coords) > 2:
                midpoint = (path_coords[0] + path_coords[1]) / 2
                ax.text(midpoint[0], midpoint[1] - 2, f"R{route_counter}: {total_distance:.1f}km", 
                        fontsize=7, color=route_color, bbox=dict(facecolor='white', alpha=0.7, pad=1))
                
            route_counter += 1

    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    # Adjust legend if too many routes
    if route_counter < 15:
       ax.legend(fontsize='small', loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)


# --- Initial Solution Heuristic ---
def savings_algorithm(cities, depots, demands, params):
    """
    Implements Clarke-Wright savings algorithm for better initial solutions.
    Returns a solution for multiple depots.
    """
    global route_cache
    route_cache = {}  # Clear cache for new problem instance
    
    solution = [[] for _ in range(len(depots))]
    assigned = set()
    num_customers = len(cities)
    
    # For each depot, create routes
    for d_idx, depot in enumerate(depots):
        # Start with individual routes (depot -> customer -> depot)
        individual_routes = []
        for c in range(num_customers):
            if c in assigned:
                continue
            
            # Check if customer can be served from this depot
            route = [c]
            route_details = calculate_route_details(route, cities, depot, demands, params)
            if route_details and route_details['feasible']:
                individual_routes.append(route)
            
        # Calculate savings for all pairs
        savings = []
        for i, route_i in enumerate(individual_routes):
            for j, route_j in enumerate(individual_routes):
                if i != j:
                    # Calculate savings: dist(0,i) + dist(0,j) - dist(i,j)
                    # where 0 is depot, i is last of route_i, j is first of route_j
                    i_last = route_i[-1]
                    j_first = route_j[0]
                    
                    dist_0_i = calculate_distance(depot, cities[i_last])
                    dist_0_j = calculate_distance(depot, cities[j_first])
                    dist_i_j = calculate_distance(cities[i_last], cities[j_first])
                    
                    saving = dist_0_i + dist_0_j - dist_i_j
                    savings.append((saving, i, j))
        
        # Sort savings in descending order
        savings.sort(reverse=True)
        
        # Merge routes using savings
        merged = [False] * len(individual_routes)
        depot_routes = []
        
        # Try merging based on savings
        for saving, i, j in savings:
            if merged[i] or merged[j]:
                continue
                
            merged_route = individual_routes[i] + individual_routes[j]
            route_details = calculate_route_details(merged_route, cities, depot, demands, params)
            
            if route_details and route_details['feasible']:
                # Successfully merged
                depot_routes.append(merged_route)
                merged[i] = merged[j] = True
        
        # Add routes that couldn't be merged
        for i, route in enumerate(individual_routes):
            if not merged[i]:
                depot_routes.append(route)
        
        # Add to solution and mark customers as assigned
        for route in depot_routes:
            solution[d_idx].append(route)
            for c in route:
                assigned.add(c)
    
    # Assign remaining customers using simple greedy insertion
    unassigned = [c for c in range(num_customers) if c not in assigned]
    if unassigned:
        for c in unassigned:
            best_cost = float('inf')
            best_insertion = None
            
            for d_idx, depot in enumerate(depots):
                for r_idx, route in enumerate(solution[d_idx]):
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [c] + route[pos:]
                        route_details = calculate_route_details(new_route, cities, depot, demands, params)
                        
                        if route_details and route_details['feasible']:
                            # Use energy consumption as cost metric
                            cost = route_details['energy']
                            if cost < best_cost:
                                best_cost = cost
                                best_insertion = (d_idx, r_idx, pos)
            
            # Insert customer at best position found
            if best_insertion:
                d_idx, r_idx, pos = best_insertion
                solution[d_idx][r_idx].insert(pos, c)
                assigned.add(c)
            else:
                # Try creating new routes for remaining customers
                for d_idx, depot in enumerate(depots):
                    if len(solution[d_idx]) < params.get('max_vehicles_per_depot', float('inf')):
                        route = [c]
                        route_details = calculate_route_details(route, cities, depot, demands, params)
                        if route_details and route_details['feasible']:
                            solution[d_idx].append(route)
                            assigned.add(c)
                            break
    
    return solution


def simple_greedy_insertion(cities, depots, demands, params):
    """Very basic greedy insertion heuristic respecting constraints."""
    num_customers = len(cities)
    num_depots = len(depots)
    max_vehicles = params.get('max_vehicles_per_depot', 1) * num_depots
    solution = [[] for _ in range(num_depots)] # List of lists of routes per depot
    assigned_customers = set()
    vehicles_used = [0] * num_depots
    customer_indices = list(range(num_customers))
    random.shuffle(customer_indices) # Process customers in random order

    for cust_idx in customer_indices:
        if cust_idx in assigned_customers: continue

        best_insertion_cost = float('inf')
        best_insertion_info = None # (depot_idx, route_list_idx, position, cost)

        # Try inserting into existing routes or starting a new route
        for d_idx in range(num_depots):
            # Try existing routes for this depot
            for r_idx, route in enumerate(solution[d_idx]):
                 # Try inserting at each possible position
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [cust_idx] + route[pos:]
                    # Check feasibility (capacity, energy)
                    route_details = calculate_route_details(new_route, cities, depots[d_idx], demands, params)
                    if route_details and route_details['feasible']:
                        # Use increase in sum of arrival times as cost proxy (simplified)
                        cost_increase = demands[cust_idx] # Very rough proxy
                        if cost_increase < best_insertion_cost:
                             best_insertion_cost = cost_increase
                             best_insertion_info = (d_idx, r_idx, pos)

            # Try starting a new route if allowed
            if vehicles_used[d_idx] < params.get('max_vehicles_per_depot', 1):
                 new_route = [cust_idx]
                 route_details = calculate_route_details(new_route, cities, depots[d_idx], demands, params)
                 if route_details and route_details['feasible']:
                    cost_increase = demands[cust_idx] * 2 # Penalize new route slightly
                    if cost_increase < best_insertion_cost:
                         best_insertion_cost = cost_increase
                         best_insertion_info = (d_idx, -1, 0) # -1 route_idx means new route

        # Perform the best insertion found
        if best_insertion_info:
            d_idx, r_idx, pos = best_insertion_info
            if r_idx == -1: # Start new route
                solution[d_idx].append([cust_idx])
                vehicles_used[d_idx] += 1
            else: # Insert into existing route
                solution[d_idx][r_idx].insert(pos, cust_idx)
            assigned_customers.add(cust_idx)
        # else: customer could not be assigned (increase vehicles or relax constraints)

    # Remove empty routes possibly created
    final_solution = [[route for route in depot_routes if route] for depot_routes in solution]
    return final_solution


# --- VNS Neighborhood Operators ---

# Relocate: Move customer `cust_idx` from route `r1` at `depot1` to route `r2` at `depot2` position `pos`.
def apply_relocate(solution, depot1, route_idx1, cust_pos1, depot2, route_idx2, pos2):
    new_solution = copy.deepcopy(solution)
    # Check bounds
    if depot1 >= len(new_solution) or route_idx1 >= len(new_solution[depot1]) or cust_pos1 >= len(new_solution[depot1][route_idx1]): return None
    if depot2 >= len(new_solution) or route_idx2 >= len(new_solution[depot2]) or pos2 > len(new_solution[depot2][route_idx2]): return None

    cust_idx = new_solution[depot1][route_idx1].pop(cust_pos1)

    # Insert into target route
    # Handle case where source and target are the same route
    if depot1 == depot2 and route_idx1 == route_idx2:
        # If inserting after the original position in the same route, adjust insertion index
        actual_pos2 = pos2 if pos2 <= cust_pos1 else pos2 -1
        if actual_pos2 > len(new_solution[depot1][route_idx1]): return None # Check adjusted bound
        new_solution[depot1][route_idx1].insert(actual_pos2, cust_idx)
    else:
        new_solution[depot2][route_idx2].insert(pos2, cust_idx)

    # Clean up empty source route if necessary
    if not new_solution[depot1][route_idx1]:
        del new_solution[depot1][route_idx1]

    return new_solution


# Exchange: Swap customer at (depot1, route_idx1, cust_pos1) with customer at (depot2, route_idx2, cust_pos2)
def apply_exchange(solution, depot1, route_idx1, cust_pos1, depot2, route_idx2, cust_pos2):
    new_solution = copy.deepcopy(solution)
    # Check bounds
    if depot1 >= len(new_solution) or route_idx1 >= len(new_solution[depot1]) or cust_pos1 >= len(new_solution[depot1][route_idx1]): return None
    if depot2 >= len(new_solution) or route_idx2 >= len(new_solution[depot2]) or cust_pos2 >= len(new_solution[depot2][route_idx2]): return None
    # Cannot exchange within the same route position
    if depot1 == depot2 and route_idx1 == route_idx2 and cust_pos1 == cust_pos2: return None

    cust1 = new_solution[depot1][route_idx1][cust_pos1]
    cust2 = new_solution[depot2][route_idx2][cust_pos2]

    new_solution[depot1][route_idx1][cust_pos1] = cust2
    new_solution[depot2][route_idx2][cust_pos2] = cust1

    return new_solution

# 2-Opt Intra-route: Apply 2-opt swap within a single route
def apply_two_opt_intra(solution, depot_idx, route_idx, i, k):
    new_solution = copy.deepcopy(solution)
    # Check bounds
    if depot_idx >= len(new_solution) or route_idx >= len(new_solution[depot_idx]): return None
    route = new_solution[depot_idx][route_idx]
    if i >= k or k >= len(route): return None # Invalid swap indices

    # Perform the 2-opt swap (reverse segment from i to k)
    segment_to_reverse = route[i : k+1]
    segment_to_reverse.reverse()
    new_solution[depot_idx][route_idx] = route[:i] + segment_to_reverse + route[k+1:]

    return new_solution


def explore_neighborhood(solution, operator_type, cities, depots, demands, params, current_objective, max_evaluations=1000):
    """Explores a neighborhood using the specified operator type (first improvement) with a limit on evaluations."""
    best_neighbor = None
    best_neighbor_objective = current_objective
    evaluations = 0
    
    # Use randomization for better exploration
    if operator_type == 'relocate':
        # Generate all possible relocate moves
        relocate_moves = []
        for d1 in range(len(solution)):
            for r1_idx, r1 in enumerate(solution[d1]):
                for c1_pos in range(len(r1)):
                    for d2 in range(len(solution)):
                        for r2_idx, r2 in enumerate(solution[d2]):
                            # Try inserting at each position in target route r2
                            for c2_pos in range(len(r2) + 1):
                                # Avoid moving to the exact same spot if routes are the same
                                if d1==d2 and r1_idx==r2_idx and c1_pos == c2_pos: continue
                                relocate_moves.append((d1, r1_idx, c1_pos, d2, r2_idx, c2_pos))
        
        # Randomize the order for better exploration
        random.shuffle(relocate_moves)
        for move in relocate_moves:
            d1, r1_idx, c1_pos, d2, r2_idx, c2_pos = move
            neighbor = apply_relocate(solution, d1, r1_idx, c1_pos, d2, r2_idx, c2_pos)
            if neighbor:
                evaluations += 1
                obj, feasible, _ = calculate_total_objective(neighbor, cities, depots, demands, params)
                if feasible and obj < best_neighbor_objective:
                    return neighbor, obj, evaluations  # First improvement
                
                if evaluations >= max_evaluations:
                    break

    elif operator_type == 'exchange':
        # Generate all possible exchange moves
        exchange_moves = []
        for d1 in range(len(solution)):
            for r1_idx, r1 in enumerate(solution[d1]):
                for c1_pos in range(len(r1)):
                    for d2 in range(len(solution)):
                        for r2_idx, r2 in enumerate(solution[d2]):
                            for c2_pos in range(len(r2)):
                                # Skip same position exchanges
                                if d1==d2 and r1_idx==r2_idx and c1_pos==c2_pos: continue
                                exchange_moves.append((d1, r1_idx, c1_pos, d2, r2_idx, c2_pos))
        
        random.shuffle(exchange_moves)
        for move in exchange_moves:
            d1, r1_idx, c1_pos, d2, r2_idx, c2_pos = move
            neighbor = apply_exchange(solution, d1, r1_idx, c1_pos, d2, r2_idx, c2_pos)
            if neighbor:
                evaluations += 1
                obj, feasible, _ = calculate_total_objective(neighbor, cities, depots, demands, params)
                if feasible and obj < best_neighbor_objective:
                    return neighbor, obj, evaluations  # First improvement
                
                if evaluations >= max_evaluations:
                    break

    elif operator_type == 'two_opt_intra':
        # Generate all possible 2-opt moves
        two_opt_moves = []
        for d in range(len(solution)):
            for r_idx, route in enumerate(solution[d]):
                if len(route) < 2: continue  # Need at least 2 customers
                for i in range(len(route) - 1):
                    for k in range(i + 1, len(route)):
                        two_opt_moves.append((d, r_idx, i, k))
        
        random.shuffle(two_opt_moves)
        for move in two_opt_moves:
            d, r_idx, i, k = move
            neighbor = apply_two_opt_intra(solution, d, r_idx, i, k)
            if neighbor:
                evaluations += 1
                obj, feasible, _ = calculate_total_objective(neighbor, cities, depots, demands, params)
                if feasible and obj < best_neighbor_objective:
                    return neighbor, obj, evaluations  # First improvement
                
                if evaluations >= max_evaluations:
                    break

    return None, current_objective, evaluations  # No improvement found


def shake_solution(solution, strength, cities, depots, demands, params):
    """Applies `strength` random feasible moves (using multiple operators) to perturb solution."""
    shaken_solution = copy.deepcopy(solution)
    moves_applied = 0
    attempts = 0
    max_attempts = strength * 10 # Limit attempts to find feasible moves

    while moves_applied < strength and attempts < max_attempts:
        attempts += 1
        
        # Choose a random move type with equal probability
        move_type = random.choice(['relocate', 'exchange', 'two_opt_intra'])
        
        if move_type == 'relocate':
            # Select random customer to move
            possible_origins = []
            for d1 in range(len(shaken_solution)):
                for r1_idx, r1 in enumerate(shaken_solution[d1]):
                    for c1_pos in range(len(r1)):
                        possible_origins.append((d1, r1_idx, c1_pos))

            if not possible_origins: continue # No customers left to move
            d1, r1_idx, c1_pos = random.choice(possible_origins)

            # Select random target route and position
            possible_targets = []
            for d2 in range(len(shaken_solution)):
                 for r2_idx, r2 in enumerate(shaken_solution[d2]):
                     for c2_pos in range(len(r2) + 1):
                          # Avoid moving to exact same spot
                          if d1 == d2 and r1_idx == r2_idx and c1_pos == c2_pos: continue
                          possible_targets.append((d2, r2_idx, c2_pos))

            if not possible_targets: continue # Cannot find target
            d2, r2_idx, c2_pos = random.choice(possible_targets)

            neighbor = apply_relocate(shaken_solution, d1, r1_idx, c1_pos, d2, r2_idx, c2_pos)
            
        elif move_type == 'exchange':
            # Select two random customers to exchange
            possible_positions = []
            for d in range(len(shaken_solution)):
                for r_idx, route in enumerate(shaken_solution[d]):
                    for c_pos in range(len(route)):
                        possible_positions.append((d, r_idx, c_pos))
            
            if len(possible_positions) < 2: continue # Need at least 2 customers
            pos1_idx = random.randint(0, len(possible_positions)-1)
            pos1 = possible_positions[pos1_idx]
            
            # Pick a different second position
            possible_positions_2 = [p for i, p in enumerate(possible_positions) if i != pos1_idx]
            if not possible_positions_2: continue
            pos2 = random.choice(possible_positions_2)
            
            neighbor = apply_exchange(shaken_solution, pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2])
            
        else: # two_opt_intra
            # Select random route
            route_positions = []
            for d in range(len(shaken_solution)):
                for r_idx, route in enumerate(shaken_solution[d]):
                    if len(route) >= 2:  # Need at least 2 customers
                        route_positions.append((d, r_idx))
            
            if not route_positions: continue
            d, r_idx = random.choice(route_positions)
            route = shaken_solution[d][r_idx]
            
            # Pick random segment to reverse
            i = random.randint(0, len(route)-2)
            k = random.randint(i+1, len(route)-1)
            
            neighbor = apply_two_opt_intra(shaken_solution, d, r_idx, i, k)

        if neighbor:
             _, feasible, _ = calculate_total_objective(neighbor, cities, depots, demands, params)
             if feasible:
                 shaken_solution = neighbor
                 moves_applied += 1

    return shaken_solution


# --- VNS Solver ---
def vns_solver(cities, depots, demands, params, plot_placeholder, status_placeholder, length_placeholder, fig, ax):
    """Performs Variable Neighborhood Search."""
    global route_cache
    route_cache = {}  # Clear cache before starting
    
    # Create progress bar
    progress_bar = st.progress(0)
    
    status_placeholder.info("Generating initial solution...")
    # Try more advanced construction heuristic (savings algorithm)
    current_solution = savings_algorithm(cities, depots, demands, params)
    current_obj, feasible, _ = calculate_total_objective(current_solution, cities, depots, demands, params)
    
    # Fallback to greedy insertion if savings method fails
    if not feasible:
        status_placeholder.warning("Advanced initial solution failed. Trying greedy insertion...")
        current_solution = simple_greedy_insertion(cities, depots, demands, params)
        current_obj, feasible, _ = calculate_total_objective(current_solution, cities, depots, demands, params)
        
    if not feasible:
        status_placeholder.error("Initial solution generation failed to find feasible solution.")
        return None, float('inf')

    best_solution = copy.deepcopy(current_solution)
    best_obj = current_obj
    
    # Display metrics with more information
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    total_customers = len(cities)
    with metrics_col1:
        best_obj_metric = st.metric("Best Objective", f"{best_obj:.2f}")
    with metrics_col2:
        current_obj_metric = st.metric("Current Objective", f"{current_obj:.2f}")
    with metrics_col3:
        # Count vehicles used
        total_vehicles = sum(len(routes) for routes in best_solution)
        vehicles_metric = st.metric("Vehicles Used", f"{total_vehicles}/{len(depots) * params['max_vehicles_per_depot']}")

    plot_vrp_solution(cities, depots, best_solution, f"Initial Solution (Obj: {best_obj:.2f})", ax)
    plot_placeholder.pyplot(fig)
    length_placeholder.metric("Current Objective", f"{current_obj:.2f}")
    time.sleep(st.session_state.vns_delay)

    iter_vns = 0
    iter_no_improve = 0
    total_evaluations = 0
    total_improvements = 0

    # Stats tracking
    stats = {
        'iter_times': [],
        'objectives': [current_obj],
        'improvements': []
    }
    
    start_time = time.time()

    while iter_vns < MAX_ITERATIONS_VNS and iter_no_improve < MAX_ITER_NO_IMPROVEMENT_VNS:
        iter_start = time.time()
        iter_vns += 1
        progress_bar.progress(iter_vns / MAX_ITERATIONS_VNS)
        k = 0 # Neighborhood index
        improvement_found_iter = False

        status_placeholder.info(f"Iteration {iter_vns}/{MAX_ITERATIONS_VNS} (No improvement: {iter_no_improve})")

        while k < len(NEIGHBORHOOD_ORDER_VNS):
            neighborhood_type = NEIGHBORHOOD_ORDER_VNS[k]

            # Explore neighborhood Nk(current_solution)
            # Use first improvement strategy
            neighbor_solution, neighbor_obj, evals = explore_neighborhood(
                current_solution, neighborhood_type, cities, depots, demands, params, current_obj
            )
            total_evaluations += evals

            if neighbor_solution: # Found an improving feasible solution
                current_solution = neighbor_solution
                current_obj = neighbor_obj
                improvement_found_iter = True
                total_improvements += 1
                stats['improvements'].append((iter_vns, k, current_obj))
                
                # Update metrics
                current_obj_metric.metric("Current Objective", f"{current_obj:.2f}")
                
                # Plot improvement
                plot_vrp_solution(cities, depots, current_solution, f"Iter {iter_vns} - N_{k} Improved (Obj: {current_obj:.2f})", ax)
                plot_placeholder.pyplot(fig)
                time.sleep(st.session_state.vns_delay)

                k = 0 # Go back to the first neighborhood
            else:
                k += 1 # Move to the next neighborhood

        # Update best solution if current is better
        if current_obj < best_obj:
             best_solution = copy.deepcopy(current_solution)
             best_obj = current_obj
             iter_no_improve = 0 # Reset counter
             
             # Update metrics
             best_obj_metric.metric("Best Objective", f"{best_obj:.2f}")
             total_vehicles = sum(len(routes) for routes in best_solution)
             vehicles_metric.metric("Vehicles Used", f"{total_vehicles}/{len(depots) * params['max_vehicles_per_depot']}")
             
             status_placeholder.info(f"Iteration {iter_vns}: **New Best Found!** Obj: {best_obj:.2f}")
        else:
            # Only increment if no improvement was found *at all* in this iteration's neighborhood search
             if not improvement_found_iter:
                 iter_no_improve += 1

        # Track stats
        iter_time = time.time() - iter_start
        stats['iter_times'].append(iter_time)
        stats['objectives'].append(min(current_obj, stats['objectives'][-1]))

        # If no improvement in any neighborhood (k went through all), shake
        if not improvement_found_iter and iter_no_improve < MAX_ITER_NO_IMPROVEMENT_VNS: # Avoid shaking on last iter if stopping due to no improvement
            status_placeholder.info(f"Iteration {iter_vns}: Shaking (No improvement: {iter_no_improve})...")
            current_solution = shake_solution(best_solution, SHAKE_STRENGTH_VNS, cities, depots, demands, params)
            current_obj, feasible, _ = calculate_total_objective(current_solution, cities, depots, demands, params)
            if not feasible:
                # If shake produces infeasible, revert to best known feasible solution
                current_solution = copy.deepcopy(best_solution)
                current_obj = best_obj
            # Update metrics after shake
            current_obj_metric.metric("Current Objective", f"{current_obj:.2f}")
            # Plot after shake
            plot_vrp_solution(cities, depots, current_solution, f"Iter {iter_vns} - After Shake (Obj: {current_obj:.2f})", ax)
            plot_placeholder.pyplot(fig)
            length_placeholder.metric("Current Objective", f"{current_obj:.2f}")
            time.sleep(st.session_state.vns_delay)

    total_time = time.time() - start_time
    status_placeholder.success(f"VNS Finished in {total_time:.2f}s. Best Objective: {best_obj:.2f}")
    
    # Display final stats
    st.write(f"Total evaluations: {total_evaluations}, Total improvements: {total_improvements}")
    st.write(f"Average time per iteration: {sum(stats['iter_times'])/len(stats['iter_times']):.4f}s")
    
    # Plot convergence graph
    fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
    ax_conv.plot(range(len(stats['objectives'])), stats['objectives'], 'b-')
    ax_conv.set_xlabel('Iteration')
    ax_conv.set_ylabel('Best Objective')
    ax_conv.set_title('Convergence History')
    ax_conv.grid(True)
    st.pyplot(fig_conv)
    
    plot_vrp_solution(cities, depots, best_solution, f"Final Best Solution (Obj: {best_obj:.2f})", ax)
    plot_placeholder.pyplot(fig)
    length_placeholder.metric("Final Objective", f"{best_obj:.2f}")

    return best_solution, best_obj


# --- Streamlit App Layout ---

st.set_page_config(layout="wide")
st.title("VNS for Multi-Depot VRP with Energy Constraints")
st.markdown("Implementing the VNS framework (Algo 1 structure) for the VRP defined in [Guo et al.](https://arxiv.org/abs/2402.15870) (Eq 1-18), using standard neighborhood exploration instead of ML/GNN operator selection.")


# --- Sidebar Controls ---
st.sidebar.header("Problem Setup")
num_customers = st.sidebar.slider("Number of Customers", 5, 50, 15)
num_depots = st.sidebar.slider("Number of Depots", 1, 5, 2)
# Use default drone params, maybe allow overrides later
st.sidebar.subheader("Drone Parameters (Defaults)")
st.sidebar.json(DEFAULT_DRONE_PARAMS, expanded=False)
params = DEFAULT_DRONE_PARAMS.copy()
# Add max vehicles per depot (simplified from Ni in Eq. 8) [cite: 95]
params['max_vehicles_per_depot'] = st.sidebar.slider("Max Vehicles per Depot", 1, 10, 3)


st.sidebar.header("VNS Controls")
st.session_state.vns_delay = st.sidebar.slider("VNS Step Delay (s)", 0.0, 2.0, 0.2, 0.05)


# --- Main Area ---
col1, col2 = st.columns([3, 1]) # Plot takes more space
with col1:
    plot_placeholder = st.empty()
with col2:
    status_placeholder = st.empty()
    length_placeholder = st.empty()
    details_placeholder = st.empty()

# --- Button Actions ---
if st.sidebar.button("Generate New Instance", key='generate_vrp'):
    # Generate customer locations and demands
    st.session_state.cities = (np.random.rand(num_customers, 2) * 100).tolist()
    # Simple demand generation (e.g., 0.1 to 1.0 units)
    st.session_state.demands = (np.random.rand(num_customers) * 0.9 + 0.1).tolist()
    # Generate depot locations (spread them out roughly)
    depot_coords = []
    for i in range(num_depots):
         depot_coords.append((random.uniform(10,90), random.uniform(10,90)))
    st.session_state.depots = depot_coords

    st.session_state.solution = None
    st.session_state.objective = float('inf')
    status_placeholder.info("Generated new VRP instance.")
    length_placeholder.text("")
    details_placeholder.text("")

    # Initial plot
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_vrp_solution(st.session_state.cities, st.session_state.depots, [], "Generated Instance", ax)
    plot_placeholder.pyplot(fig)


if st.sidebar.button("Run VNS Solver", key='solve_vns'):
    if 'cities' not in st.session_state:
        st.sidebar.error("Please generate an instance first.")
    else:
        status_placeholder.info("Starting VNS Solver...")
        length_placeholder.text("")
        details_placeholder.text("")
        fig, ax = plt.subplots(figsize=(10, 7)) # Create figure/axis for plotting

        # Run the VNS solver
        start_time = time.time()
        best_solution, best_objective = vns_solver(
            st.session_state.cities,
            st.session_state.depots,
            st.session_state.demands,
            params, # Pass drone and vehicle params
            plot_placeholder,
            status_placeholder,
            length_placeholder,
            fig,
            ax
        )
        end_time = time.time()

        st.session_state.solution = best_solution
        st.session_state.objective = best_objective

        if best_solution is not None:
             status_placeholder.success(f"VNS finished in {end_time - start_time:.2f} seconds.")
             length_placeholder.metric("Best Objective Found", f"{best_objective:.2f}")
             # Display route details
             _, _, final_details = calculate_total_objective(best_solution, st.session_state.cities, st.session_state.depots, st.session_state.demands, params)
             details_str = "Final Route Details:\n"
             route_count = 0
             for d_idx, depot_routes in enumerate(final_details):
                 details_str += f"Depot {d_idx}:\n"
                 for r_idx, r_details in enumerate(depot_routes):
                      route_indices = best_solution[d_idx][r_idx]
                      if r_details['feasible']:
                          details_str += f"  Route {route_count}: {' -> '.join(map(str, route_indices))}\n"
                          details_str += f"    Dist: {r_details['distance']:.2f} km, Energy: {r_details['energy']:.2f} Wh, SumTimes: {sum(r_details['arrival_times'].values()):.2f} s\n"
                      else:
                          details_str += f"  Route {route_count} (Infeasible): {r_details.get('infeasibility_reason', 'Unknown')}\n"
                      route_count+=1

             details_placeholder.text_area("Route Details", details_str, height=200)

        else:
             status_placeholder.error("VNS failed to find a feasible solution.")


# --- Display Initial/Last State ---
if 'cities' in st.session_state:
    fig, ax = plt.subplots(figsize=(10, 7))
    title = "Current VRP Instance"
    if 'solution' in st.session_state and st.session_state.solution is not None:
        title = f"Last Solution (Obj: {st.session_state.objective:.2f})"
        plot_vrp_solution(st.session_state.cities, st.session_state.depots, st.session_state.solution, title, ax)
        length_placeholder.metric("Last Objective", f"{st.session_state.objective:.2f}")

    else:
         plot_vrp_solution(st.session_state.cities, st.session_state.depots, [], title, ax)
    plot_placeholder.pyplot(fig)

else:
    status_placeholder.info("Click 'Generate New Instance' in the sidebar.")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("No Instance Generated")
    ax.set_xticks([])
    ax.set_yticks([])
    plot_placeholder.pyplot(fig)