# streamlit_vns_vrp_energy.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time

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

# --- Helper Functions ---

def calculate_distance(p1, p2):
    """Calculate Euclidean distance."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_energy_consumption(dist, payload, height, params):
    """Calculates energy based on Eq. 1 from the paper."""
    # R_ij^k = gamma * H_ij^k * (w0 + w_ij^k) + rho * L_ij * (w0 + w_ij^k)
    # Simplified: assume H is constant, payload w_ij^k is average for segment for simplicity
    # A more accurate calculation would track payload changes *within* the segment.
    # Using payload *at start* of segment as approximation.
    energy = (params['gamma'] * height + params['rho'] * dist) * (params['w0'] + payload) # [cite: 85]
    return energy

def calculate_route_details(route_indices, cities, depot_coord, demands, params):
    """
    Calculates total distance, arrival times, payload, energy consumption, and feasibility for a single route.
    Returns details dict or None if infeasible.
    """
    if not route_indices:
        return {'distance': 0, 'arrival_times': {}, 'total_time': 0, 'energy': 0, 'feasible': True, 'infeasibility_reason': None}

    total_dist = 0
    current_payload = sum(demands[i] for i in route_indices)
    current_energy = params['E']
    current_time = 0
    last_coord = depot_coord
    details = {'arrival_times': {}, 'total_time': 0, 'max_payload_violation': False, 'energy_violation': False}

    # Check initial capacity
    if current_payload > params['capacity']:
        details['max_payload_violation'] = True
        return {'feasible': False, 'infeasibility_reason': f'Initial payload {current_payload:.2f} exceeds capacity {params["capacity"]:.2f}'}

    route_path = [depot_coord] + [cities[i] for i in route_indices] + [depot_coord]

    # Fly to first customer
    dist_to_first = calculate_distance(depot_coord, cities[route_indices[0]])
    energy_needed = calculate_energy_consumption(dist_to_first, current_payload, params['H'], params)
    if energy_needed > current_energy:
        return {'feasible': False, 'infeasibility_reason': f'Not enough energy for first leg ({energy_needed:.2f} > {current_energy:.2f})'}
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
             return {'feasible': False, 'infeasibility_reason': f'Energy fail at C{idx_u}->C{idx_v} ({energy_needed:.2f} > {current_energy:.2f})'}
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
        return {'feasible': False, 'infeasibility_reason': f'Energy fail C{idx_last}->Depot ({energy_needed:.2f} > {current_energy:.2f})'}
    current_energy -= energy_needed
    total_dist += dist_to_depot
    # Don't add time for return to depot to objective as per Eq. 2 (sum arrival times at customers) [cite: 90]

    details['distance'] = total_dist
    details['energy'] = params['E'] - current_energy # Total consumed
    details['total_time'] = current_time # Last arrival time
    details['feasible'] = True
    details['infeasibility_reason'] = None
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

    # Plot routes
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
            route_counter += 1

    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    # Adjust legend if too many routes
    if route_counter < 15:
       ax.legend(fontsize='small', loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)


# --- Initial Solution Heuristic ---
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
                        # This is not perfect, a better heuristic would use Eq 2 directly
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


def explore_neighborhood(solution, operator_type, cities, depots, demands, params, current_objective):
    """Explores a neighborhood using the specified operator type (first improvement)."""
    best_neighbor = None
    best_neighbor_objective = current_objective

    if operator_type == 'relocate':
        # Iterate through all customers and all possible insertion points
        for d1 in range(len(solution)):
            for r1_idx, r1 in enumerate(solution[d1]):
                for c1_pos in range(len(r1)):
                    for d2 in range(len(solution)):
                        for r2_idx, r2 in enumerate(solution[d2]):
                            # Try inserting at each position in target route r2
                            for c2_pos in range(len(r2) + 1):
                                # Avoid moving to the exact same spot if routes are the same
                                if d1==d2 and r1_idx==r2_idx and c1_pos == c2_pos: continue

                                neighbor = apply_relocate(solution, d1, r1_idx, c1_pos, d2, r2_idx, c2_pos)
                                if neighbor:
                                    obj, feasible, _ = calculate_total_objective(neighbor, cities, depots, demands, params)
                                    if feasible and obj < best_neighbor_objective:
                                        # Found improving feasible neighbor - return immediately (first improvement)
                                        return neighbor, obj
                        # Also try inserting into a potential new route for depot d2 (if allowed)
                        # (Simplified: Not implementing adding new routes in local search here)

    elif operator_type == 'exchange':
         # Iterate through all pairs of customers
         cust_locations = [] # Store (cust_idx, d, r, c_pos)
         for d in range(len(solution)):
             for r_idx, r in enumerate(solution[d]):
                 for c_pos in range(len(r)):
                      cust_locations.append({'idx':r[c_pos], 'd':d, 'r':r_idx, 'c_pos':c_pos})

         for i in range(len(cust_locations)):
             for j in range(i + 1, len(cust_locations)):
                 loc1 = cust_locations[i]
                 loc2 = cust_locations[j]
                 neighbor = apply_exchange(solution, loc1['d'], loc1['r'], loc1['c_pos'], loc2['d'], loc2['r'], loc2['c_pos'])
                 if neighbor:
                     obj, feasible, _ = calculate_total_objective(neighbor, cities, depots, demands, params)
                     if feasible and obj < best_neighbor_objective:
                         return neighbor, obj # First improvement

    elif operator_type == 'two_opt_intra':
        # Iterate through all routes and all possible 2-opt swaps within them
        for d in range(len(solution)):
            for r_idx, r in enumerate(solution[d]):
                 if len(r) < 2: continue # Need at least 2 edges to swap
                 for i in range(len(r) - 1):
                     for k in range(i + 1, len(r)):
                         neighbor = apply_two_opt_intra(solution, d, r_idx, i, k)
                         if neighbor:
                             obj, feasible, _ = calculate_total_objective(neighbor, cities, depots, demands, params)
                             if feasible and obj < best_neighbor_objective:
                                  return neighbor, obj # First improvement


    return None, current_objective # No improvement found


def shake_solution(solution, strength, cities, depots, demands, params):
    """Applies `strength` random feasible moves (e.g., relocate) to perturb solution."""
    shaken_solution = copy.deepcopy(solution)
    moves_applied = 0
    attempts = 0
    max_attempts = strength * 10 # Limit attempts to find feasible moves

    while moves_applied < strength and attempts < max_attempts:
        attempts += 1
        # Choose a random move type, e.g., relocate
        # (Could add random exchange etc.)

        # Select random customer to move
        possible_origins = []
        for d1 in range(len(shaken_solution)):
            for r1_idx, r1 in enumerate(shaken_solution[d1]):
                for c1_pos in range(len(r1)):
                    possible_origins.append((d1, r1_idx, c1_pos))

        if not possible_origins: break # No customers left to move
        d1, r1_idx, c1_pos = random.choice(possible_origins)

        # Select random target route and position
        possible_targets = []
        for d2 in range(len(shaken_solution)):
             for r2_idx, r2 in enumerate(shaken_solution[d2]):
                 for c2_pos in range(len(r2) + 1):
                      # Avoid moving to exact same spot
                      if d1 == d2 and r1_idx == r2_idx and c1_pos == c2_pos: continue
                      possible_targets.append((d2, r2_idx, c2_pos))
        # Add possibility of creating new route if allowed (simplified: not implemented here)

        if not possible_targets: continue # Cannot find target
        d2, r2_idx, c2_pos = random.choice(possible_targets)

        neighbor = apply_relocate(shaken_solution, d1, r1_idx, c1_pos, d2, r2_idx, c2_pos)

        if neighbor:
             _, feasible, _ = calculate_total_objective(neighbor, cities, depots, demands, params)
             if feasible:
                 shaken_solution = neighbor
                 moves_applied += 1

    # st.write(f"Shake applied {moves_applied} moves after {attempts} attempts.")
    return shaken_solution


# --- VNS Solver ---
def vns_solver(cities, depots, demands, params, plot_placeholder, status_placeholder, length_placeholder, fig, ax):
    """Performs Variable Neighborhood Search."""
    status_placeholder.info("Generating initial solution...")
    current_solution = simple_greedy_insertion(cities, depots, demands, params)
    current_obj, feasible, _ = calculate_total_objective(current_solution, cities, depots, demands, params)
    if not feasible:
        status_placeholder.error("Initial solution generation failed to find feasible solution.")
        return None, float('inf')

    best_solution = copy.deepcopy(current_solution)
    best_obj = current_obj

    plot_vrp_solution(cities, depots, best_solution, f"Initial Solution (Obj: {best_obj:.2f})", ax)
    plot_placeholder.pyplot(fig)
    length_placeholder.metric("Current Objective", f"{current_obj:.2f}")
    time.sleep(st.session_state.vns_delay)

    iter_vns = 0
    iter_no_improve = 0

    while iter_vns < MAX_ITERATIONS_VNS and iter_no_improve < MAX_ITER_NO_IMPROVEMENT_VNS:
        iter_vns += 1
        k = 0 # Neighborhood index
        improvement_found_iter = False

        status_placeholder.info(f"Iteration {iter_vns}/{MAX_ITERATIONS_VNS} (No improvement: {iter_no_improve})")

        while k < len(NEIGHBORHOOD_ORDER_VNS):
            neighborhood_type = NEIGHBORHOOD_ORDER_VNS[k]
            # st.write(f"Iter {iter_vns}, Exploring N_{k}: {neighborhood_type}")

            # Explore neighborhood Nk(current_solution)
            # Use first improvement strategy
            neighbor_solution, neighbor_obj = explore_neighborhood(
                current_solution, neighborhood_type, cities, depots, demands, params, current_obj
            )

            if neighbor_solution: # Found an improving feasible solution
                current_solution = neighbor_solution
                current_obj = neighbor_obj
                improvement_found_iter = True
                # st.write(f"  -> Improvement found! New Obj: {current_obj:.2f}. Resetting to N_0.")
                length_placeholder.metric("Current Objective", f"{current_obj:.2f}")
                plot_vrp_solution(cities, depots, current_solution, f"Iter {iter_vns} - N_{k} Improved (Obj: {current_obj:.2f})", ax)
                plot_placeholder.pyplot(fig)
                time.sleep(st.session_state.vns_delay)

                k = 0 # Go back to the first neighborhood
            else:
                # st.write(f"  -> No improvement in N_{k}.")
                k += 1 # Move to the next neighborhood

        # Update best solution if current is better
        if current_obj < best_obj:
             best_solution = copy.deepcopy(current_solution)
             best_obj = current_obj
             iter_no_improve = 0 # Reset counter
             status_placeholder.info(f"Iteration {iter_vns}: **New Best Found!** Obj: {best_obj:.2f}")
             # Maybe add another plot update for new best?
        else:
            # Only increment if no improvement was found *at all* in this iteration's neighborhood search
             if not improvement_found_iter:
                 iter_no_improve += 1


        # If no improvement in any neighborhood (k went through all), shake
        if not improvement_found_iter and iter_no_improve < MAX_ITER_NO_IMPROVEMENT_VNS : # Avoid shaking on last iter if stopping due to no improvement
            # st.write(f"Iter {iter_vns}: No improvement in any N_k. Shaking...")
            status_placeholder.info(f"Iteration {iter_vns}: Shaking (No improvement: {iter_no_improve})...")
            current_solution = shake_solution(best_solution, SHAKE_STRENGTH_VNS, cities, depots, demands, params)
            current_obj, feasible, _ = calculate_total_objective(current_solution, cities, depots, demands, params)
            if not feasible:
                # If shake produces infeasible, revert to best known feasible solution
                # st.warning("Shake resulted in infeasible solution, reverting to best.")
                current_solution = copy.deepcopy(best_solution)
                current_obj = best_obj
            # Plot after shake
            plot_vrp_solution(cities, depots, current_solution, f"Iter {iter_vns} - After Shake (Obj: {current_obj:.2f})", ax)
            plot_placeholder.pyplot(fig)
            length_placeholder.metric("Current Objective", f"{current_obj:.2f}")
            time.sleep(st.session_state.vns_delay)


    status_placeholder.success(f"VNS Finished. Best Objective: {best_obj:.2f}")
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