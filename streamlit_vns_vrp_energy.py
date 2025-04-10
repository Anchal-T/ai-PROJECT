import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time

# --- Constants & Parameters ---
DEFAULT_DRONE_PARAMS = {
    'capacity': 10.0,       # Max payload (kg)
    'w0': 10.0,             # Drone curb weight (kg)
    'gamma': 2.0,          # Energy for takeoff/landing (Wh/kg/km)
    'rho': 3.5,            # Energy for flight (Wh/kg/km)
    'H': 0.05,             # Fixed flight height (km)
    'E': 1000.0,            # Max battery energy (Wh)
    'speed': 40.0          # Average speed (km/h)
}
MAX_ITERATIONS_VNS = 50
MAX_ITER_NO_IMPROVEMENT_VNS = 15
NEIGHBORHOOD_ORDER_VNS = ['relocate', 'exchange', 'two_opt_intra']
SHAKE_STRENGTH_VNS = 3
route_cache = {}

# --- Helper Functions ---

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_energy_consumption(dist, payload, height, params):
    """Calculate energy consumption based on distance, payload, and height."""
    return (params['gamma'] * height + params['rho'] * dist) * (params['w0'] + payload)

def get_route_key(route_indices):
    """Generate a hashable key for route caching."""
    return tuple(route_indices)

def calculate_route_details(route_indices, cities, depot_coord, demands, params):
    """Calculate route details including distance, energy, and feasibility."""
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
    details = {'arrival_times': {}, 'total_time': 0}

    if current_payload > params['capacity']:
        result = {'feasible': False, 'infeasibility_reason': f'Payload {current_payload:.2f} > {params["capacity"]:.2f}'}
        route_cache[route_key] = result
        return result

    route_path = [depot_coord] + [cities[i] for i in route_indices] + [depot_coord]
    for i in range(len(route_path) - 1):
        dist = calculate_distance(route_path[i], route_path[i + 1])
        energy_needed = calculate_energy_consumption(dist, current_payload, params['H'], params)
        if energy_needed > current_energy:
            result = {'feasible': False, 'infeasibility_reason': f'Energy needed {energy_needed:.2f} > {current_energy:.2f}'}
            route_cache[route_key] = result
            return result
        current_energy -= energy_needed
        total_dist += dist
        time_taken = dist / params['speed'] * 3600  # Convert to seconds
        current_time += time_taken
        if i < len(route_indices):
            details['arrival_times'][route_indices[i]] = current_time
        if i < len(route_indices):
            current_payload -= demands[route_indices[i]]

    details['distance'] = total_dist
    details['energy'] = params['E'] - current_energy
    details['total_time'] = current_time
    details['feasible'] = True
    route_cache[route_key] = details
    return details

def calculate_total_objective(solution, cities, depots, demands, params):
    """Compute total objective (sum of arrival times) and check feasibility."""
    total_sum_arrival_times = 0
    all_feasible = True
    max_vehicles_per_depot = params.get('max_vehicles_per_depot', 100)
    details_list = []

    for depot_idx, depot_routes in enumerate(solution):
        if len(depot_routes) > max_vehicles_per_depot:
            all_feasible = False
            details_list.append([{'feasible': False, 'infeasibility_reason': f'Too many vehicles: {len(depot_routes)} > {max_vehicles_per_depot}'}] * len(depot_routes))
            continue
        depot_details = []
        depot_coord = depots[depot_idx]
        for route_indices in depot_routes:
            route_details = calculate_route_details(route_indices, cities, depot_coord, demands, params)
            if not route_details['feasible']:
                all_feasible = False
            else:
                total_sum_arrival_times += sum(route_details['arrival_times'].values())
            depot_details.append(route_details)
        details_list.append(depot_details)

    return total_sum_arrival_times if all_feasible else float('inf'), all_feasible, details_list

def plot_vrp_solution(cities, depots, solution, title, ax, demands, params):
    """Visualize the VRP solution with routes colored by feasibility."""
    ax.clear()
    colors = plt.cm.get_cmap('tab10', len(depots) + sum(len(routes) for routes in solution))
    customer_coords = np.array(cities)
    ax.plot(customer_coords[:, 0], customer_coords[:, 1], 'ko', label='Customers')
    for i, city in enumerate(cities):
        ax.text(city[0] + 0.5, city[1] + 0.5, str(i), fontsize=8)
    depot_coords = np.array(depots)
    ax.plot(depot_coords[:, 0], depot_coords[:, 1], 'rs', label='Depots')
    for i, depot in enumerate(depots):
        ax.text(depot[0] + 0.5, depot[1] + 0.5, f"D{i}", fontsize=9, color='red')

    route_counter = 0
    for depot_idx, depot_routes in enumerate(solution):
        depot_coord = depots[depot_idx]
        for route_indices in depot_routes:
            if not route_indices:
                continue
            details = calculate_route_details(route_indices, cities, depot_coord, demands, params)
            route_color = 'red' if not details['feasible'] else colors(route_counter % colors.N)
            path_coords = [depot_coord] + [cities[i] for i in route_indices] + [depot_coord]
            path_coords = np.array(path_coords)
            ax.plot(path_coords[:, 0], path_coords[:, 1], '-', color=route_color, lw=1.5, alpha=0.8, label=f"D{depot_idx}-R{route_counter}")
            total_distance = sum(calculate_distance(path_coords[i], path_coords[i + 1]) for i in range(len(path_coords) - 1))
            if len(path_coords) > 2:
                midpoint = (path_coords[0] + path_coords[1]) / 2
                ax.text(midpoint[0], midpoint[1] - 2, f"R{route_counter}: {total_distance:.1f}km", fontsize=7, color=route_color, bbox=dict(facecolor='white', alpha=0.7))
            route_counter += 1

    ax.set_title(title)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    if route_counter < 15:
        ax.legend(fontsize='small', loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

# --- Initial Solution Heuristic ---

def savings_algorithm(cities, depots, demands, params):
    """Generate an initial solution using the Clarke-Wright savings algorithm."""
    num_customers = len(cities)
    num_depots = len(depots)
    solution = [[] for _ in range(num_depots)]
    assigned = set()

    for d_idx, depot in enumerate(depots):
        individual_routes = []
        for c in range(num_customers):
            if c not in assigned:
                route = [c]
                if calculate_route_details(route, cities, depot, demands, params)['feasible']:
                    individual_routes.append(route)

        savings = []
        for i, route_i in enumerate(individual_routes):
            for j, route_j in enumerate(individual_routes):
                if i != j:
                    i_last, j_first = route_i[-1], route_j[0]
                    dist_0_i = calculate_distance(depot, cities[i_last])
                    dist_0_j = calculate_distance(depot, cities[j_first])
                    dist_i_j = calculate_distance(cities[i_last], cities[j_first])
                    saving = dist_0_i + dist_0_j - dist_i_j
                    savings.append((saving, i, j))
        savings.sort(reverse=True)

        merged = [False] * len(individual_routes)
        depot_routes = []
        for saving, i, j in savings:
            if merged[i] or merged[j]:
                continue
            merged_route = individual_routes[i] + individual_routes[j]
            if calculate_route_details(merged_route, cities, depot, demands, params)['feasible']:
                depot_routes.append(merged_route)
                merged[i] = merged[j] = True
        for i, route in enumerate(individual_routes):
            if not merged[i]:
                depot_routes.append(route)

        for route in depot_routes:
            solution[d_idx].append(route)
            assigned.update(route)

    unassigned = [c for c in range(num_customers) if c not in assigned]
    for c in unassigned:
        for d_idx, depot in enumerate(depots):
            if len(solution[d_idx]) < params.get('max_vehicles_per_depot', float('inf')):
                route = [c]
                if calculate_route_details(route, cities, depot, demands, params)['feasible']:
                    solution[d_idx].append(route)
                    assigned.add(c)
                    break
    return solution

def simple_greedy_insertion(cities, depots, demands, params):
    """Generate an initial solution using a greedy insertion heuristic."""
    num_customers = len(cities)
    num_depots = len(depots)
    solution = [[] for _ in range(num_depots)]
    assigned = set()
    customer_indices = list(range(num_customers))
    random.shuffle(customer_indices)

    for cust_idx in customer_indices:
        if cust_idx in assigned:
            continue
        best_cost = float('inf')
        best_insertion = None
        for d_idx in range(num_depots):
            for r_idx, route in enumerate(solution[d_idx]):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [cust_idx] + route[pos:]
                    details = calculate_route_details(new_route, cities, depots[d_idx], demands, params)
                    if details['feasible']:
                        cost = details['energy']
                        if cost < best_cost:
                            best_cost = cost
                            best_insertion = (d_idx, r_idx, pos)
            if len(solution[d_idx]) < params.get('max_vehicles_per_depot', 1):
                new_route = [cust_idx]
                details = calculate_route_details(new_route, cities, depots[d_idx], demands, params)
                if details['feasible'] and details['energy'] < best_cost:
                    best_cost = details['energy']
                    best_insertion = (d_idx, -1, 0)
        if best_insertion:
            d_idx, r_idx, pos = best_insertion
            if r_idx == -1:
                solution[d_idx].append([cust_idx])
            else:
                solution[d_idx][r_idx].insert(pos, cust_idx)
            assigned.add(cust_idx)
    return [[route for route in depot_routes if route] for depot_routes in solution]

# --- VNS Neighborhood Operators ---

def apply_relocate(solution, depot1, route_idx1, cust_pos1, depot2, route_idx2, pos2):
    """Relocate a customer from one route to another."""
    new_solution = copy.deepcopy(solution)
    if (depot1 >= len(new_solution) or route_idx1 >= len(new_solution[depot1]) or cust_pos1 >= len(new_solution[depot1][route_idx1]) or
        depot2 >= len(new_solution) or route_idx2 >= len(new_solution[depot2]) or pos2 > len(new_solution[depot2][route_idx2])):
        return None
    cust_idx = new_solution[depot1][route_idx1].pop(cust_pos1)
    if depot1 == depot2 and route_idx1 == route_idx2:
        actual_pos2 = pos2 if pos2 <= cust_pos1 else pos2 - 1
        new_solution[depot1][route_idx1].insert(actual_pos2, cust_idx)
    else:
        new_solution[depot2][route_idx2].insert(pos2, cust_idx)
    if not new_solution[depot1][route_idx1]:
        del new_solution[depot1][route_idx1]
    return new_solution

def apply_exchange(solution, depot1, route_idx1, cust_pos1, depot2, route_idx2, cust_pos2):
    """Exchange two customers between routes."""
    new_solution = copy.deepcopy(solution)
    if (depot1 >= len(new_solution) or route_idx1 >= len(new_solution[depot1]) or cust_pos1 >= len(new_solution[depot1][route_idx1]) or
        depot2 >= len(new_solution) or route_idx2 >= len(new_solution[depot2]) or cust_pos2 >= len(new_solution[depot2][route_idx2]) or
        (depot1 == depot2 and route_idx1 == route_idx2 and cust_pos1 == cust_pos2)):
        return None
    new_solution[depot1][route_idx1][cust_pos1], new_solution[depot2][route_idx2][cust_pos2] = (
        new_solution[depot2][route_idx2][cust_pos2], new_solution[depot1][route_idx1][cust_pos1])
    return new_solution

def apply_two_opt_intra(solution, depot_idx, route_idx, i, k):
    """Perform a 2-opt swap within a single route."""
    new_solution = copy.deepcopy(solution)
    if (depot_idx >= len(new_solution) or route_idx >= len(new_solution[depot_idx]) or
        i >= k or k >= len(new_solution[depot_idx][route_idx])):
        return None
    route = new_solution[depot_idx][route_idx]
    new_solution[depot_idx][route_idx] = route[:i] + route[i:k + 1][::-1] + route[k + 1:]
    return new_solution

def explore_neighborhood(solution, operator_type, cities, depots, demands, params, current_objective, max_evaluations=5000):
    """Explore a neighborhood to find an improving feasible solution."""
    evaluations = 0
    moves = []
    if operator_type == 'relocate':
        for d1 in range(len(solution)):
            for r1_idx, r1 in enumerate(solution[d1]):
                for c1_pos in range(len(r1)):
                    for d2 in range(len(solution)):
                        for r2_idx, r2 in enumerate(solution[d2]):
                            for c2_pos in range(len(r2) + 1):
                                if d1 == d2 and r1_idx == r2_idx and c1_pos == c2_pos:
                                    continue
                                moves.append((apply_relocate, (d1, r1_idx, c1_pos, d2, r2_idx, c2_pos)))
    elif operator_type == 'exchange':
        for d1 in range(len(solution)):
            for r1_idx, r1 in enumerate(solution[d1]):
                for c1_pos in range(len(r1)):
                    for d2 in range(len(solution)):
                        for r2_idx, r2 in enumerate(solution[d2]):
                            for c2_pos in range(len(r2)):
                                if d1 == d2 and r1_idx == r2_idx and c1_pos == c2_pos:
                                    continue
                                moves.append((apply_exchange, (d1, r1_idx, c1_pos, d2, r2_idx, c2_pos)))
    elif operator_type == 'two_opt_intra':
        for d in range(len(solution)):
            for r_idx, route in enumerate(solution[d]):
                if len(route) < 2:
                    continue
                for i in range(len(route) - 1):
                    for k in range(i + 1, len(route)):
                        moves.append((apply_two_opt_intra, (d, r_idx, i, k)))

    random.shuffle(moves)
    for apply_func, args in moves[:max_evaluations]:
        neighbor = apply_func(solution, *args)
        if neighbor:
            evaluations += 1
            obj, feasible, _ = calculate_total_objective(neighbor, cities, depots, demands, params)
            if feasible and obj < current_objective:
                return neighbor, obj, evaluations
    return None, current_objective, evaluations

def shake_solution(solution, strength, cities, depots, demands, params):
    """Perturb the solution with random feasible moves."""
    shaken_solution = copy.deepcopy(solution)
    moves_applied = 0
    attempts = 0
    max_attempts = strength * 10

    while moves_applied < strength and attempts < max_attempts:
        attempts += 1
        move_type = random.choice(['relocate', 'exchange', 'two_opt_intra'])
        if move_type == 'relocate':
            origins = [(d1, r1_idx, c1_pos) for d1 in range(len(shaken_solution)) for r1_idx, r1 in enumerate(shaken_solution[d1]) for c1_pos in range(len(r1))]
            targets = [(d2, r2_idx, c2_pos) for d2 in range(len(shaken_solution)) for r2_idx, r2 in enumerate(shaken_solution[d2]) for c2_pos in range(len(r2) + 1)]
            if not origins or not targets:
                continue
            d1, r1_idx, c1_pos = random.choice(origins)
            d2, r2_idx, c2_pos = random.choice([(d, r, p) for d, r, p in targets if not (d == d1 and r == r1_idx and p == c1_pos)])
            neighbor = apply_relocate(shaken_solution, d1, r1_idx, c1_pos, d2, r2_idx, c2_pos)
        elif move_type == 'exchange':
            positions = [(d, r_idx, c_pos) for d in range(len(shaken_solution)) for r_idx, r in enumerate(shaken_solution[d]) for c_pos in range(len(r))]
            if len(positions) < 2:
                continue
            pos1, pos2 = random.sample(positions, 2)
        else:
            routes = [(d, r_idx) for d in range(len(shaken_solution)) for r_idx, r in enumerate(shaken_solution[d]) if len(r) >= 2]
            if not routes:
                continue
            d, r_idx = random.choice(routes)
            route = shaken_solution[d][r_idx]
            i, k = sorted(random.sample(range(len(route)), 2))
            neighbor = apply_two_opt_intra(shaken_solution, d, r_idx, i, k)

            if neighbor:
                _, feasible, _ = calculate_total_objective(neighbor, cities, depots, demands, params)
                if feasible:
                    shaken_solution = neighbor
                    moves_applied += 1
    return shaken_solution

# --- VNS Solver ---

def vns_solver(cities, depots, demands, params, plot_placeholder, status_placeholder, length_placeholder, fig, ax):
    """Execute Variable Neighborhood Search to optimize the VRP solution."""
    global route_cache
    route_cache = {}
    progress_bar = st.progress(0)
    status_placeholder.info("Generating initial solution...")

    current_solution = savings_algorithm(cities, depots, demands, params)
    current_obj, feasible, _ = calculate_total_objective(current_solution, cities, depots, demands, params)
    if not feasible:
        status_placeholder.warning("Savings algorithm failed. Using greedy insertion...")
        current_solution = simple_greedy_insertion(cities, depots, demands, params)
        current_obj, feasible, _ = calculate_total_objective(current_solution, cities, depots, demands, params)
    if not feasible:
        status_placeholder.error("Failed to generate a feasible initial solution.")
        return None, float('inf')

    best_solution = copy.deepcopy(current_solution)
    best_obj = current_obj

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        best_obj_metric = st.metric("Best Objective", f"{best_obj:.2f}")
    with metrics_col2:
        current_obj_metric = st.metric("Current Objective", f"{current_obj:.2f}")
    with metrics_col3:
        vehicles_metric = st.metric("Vehicles Used", f"{sum(len(routes) for routes in best_solution)}/{len(depots) * params['max_vehicles_per_depot']}")

    iter_vns = 0
    iter_no_improve = 0
    total_evaluations = 0
    stats = {'iter_times': [], 'objectives': [current_obj], 'improvements': []}
    start_time = time.time()

    while iter_vns < MAX_ITERATIONS_VNS and iter_no_improve < MAX_ITER_NO_IMPROVEMENT_VNS:
        iter_start = time.time()
        iter_vns += 1
        progress_bar.progress(iter_vns / MAX_ITERATIONS_VNS)
        status_placeholder.info(f"Iteration {iter_vns}/{MAX_ITERATIONS_VNS} (No improvement: {iter_no_improve})")
        k = 0
        while k < len(NEIGHBORHOOD_ORDER_VNS):
            neighbor_solution, neighbor_obj, evals = explore_neighborhood(
                current_solution, NEIGHBORHOOD_ORDER_VNS[k], cities, depots, demands, params, current_obj)
            total_evaluations += evals
            if neighbor_solution:
                current_solution = neighbor_solution
                current_obj = neighbor_obj
                current_obj_metric.value = f"{current_obj:.2f}"
                plot_vrp_solution(cities, depots, current_solution, f"Iter {iter_vns} - N_{k} Improved (Obj: {current_obj:.2f})", ax, demands, params)
                plot_placeholder.pyplot(fig)
                k = 0
                if current_obj < best_obj:
                    best_solution = copy.deepcopy(current_solution)
                    best_obj = current_obj
                    best_obj_metric.value = f"{best_obj:.2f}"
                    vehicles_metric.value = f"{sum(len(routes) for routes in best_solution)}/{len(depots) * params['max_vehicles_per_depot']}"
                    status_placeholder.info(f"Iteration {iter_vns}: **New Best Found!** Obj: {best_obj:.2f}")
                    iter_no_improve = 0
                else:
                    iter_no_improve += 1
                break
            else:
                k += 1
        if k == len(NEIGHBORHOOD_ORDER_VNS) and iter_no_improve < MAX_ITER_NO_IMPROVEMENT_VNS:
            status_placeholder.info(f"Iteration {iter_vns}: Shaking...")
            current_solution = shake_solution(current_solution, SHAKE_STRENGTH_VNS, cities, depots, demands, params)
            current_obj, feasible, _ = calculate_total_objective(current_solution, cities, depots, demands, params)
            if not feasible:
                current_solution = copy.deepcopy(best_solution)
                current_obj = best_obj
            current_obj_metric.value = f"{current_obj:.2f}"
            plot_vrp_solution(cities, depots, current_solution, f"Iter {iter_vns} - After Shake (Obj: {current_obj:.2f})", ax, demands, params)
            plot_placeholder.pyplot(fig)
            iter_no_improve += 1

        stats['iter_times'].append(time.time() - iter_start)
        stats['objectives'].append(min(current_obj, stats['objectives'][-1]))

    total_time = time.time() - start_time
    status_placeholder.success(f"VNS Finished in {total_time:.2f}s. Best Objective: {best_obj:.2f}")
    st.write(f"Total evaluations: {total_evaluations}")
    st.write(f"Average time per iteration: {sum(stats['iter_times'])/len(stats['iter_times']):.4f}s")

    fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
    ax_conv.plot(range(len(stats['objectives'])), stats['objectives'], 'b-')
    ax_conv.set_xlabel('Iteration')
    ax_conv.set_ylabel('Best Objective')
    ax_conv.set_title('Convergence History')
    ax_conv.grid(True)
    st.pyplot(fig_conv)

    plot_vrp_solution(cities, depots, best_solution, f"Final Best Solution (Obj: {best_obj:.2f})", ax, demands, params)
    plot_placeholder.pyplot(fig)
    length_placeholder.metric("Final Objective", f"{best_obj:.2f}")
    return best_solution, best_obj

# --- Streamlit App Layout ---

st.set_page_config(layout="wide")
st.title("VNS for Multi-Depot VRP with Energy Constraints")
# st.markdown("Implementing VNS for the VRP defined in [Guo et al.](https://arxiv.org/abs/2402.15870).")

# Sidebar Controls
st.sidebar.header("Problem Setup")
num_customers = st.sidebar.slider("Number of Customers", 5, 50, 50)
num_depots = st.sidebar.slider("Number of Depots", 1, 5, 5)
params = DEFAULT_DRONE_PARAMS.copy()
params['max_vehicles_per_depot'] = st.sidebar.slider("Max Vehicles per Depot", 1, 10, 10)
st.sidebar.subheader("Drone Parameters")
st.sidebar.json(DEFAULT_DRONE_PARAMS, expanded=False)

# Main Area
col1, col2 = st.columns([3, 1])
with col1:
    plot_placeholder = st.empty()
with col2:
    status_placeholder = st.empty()
    length_placeholder = st.empty()
    details_placeholder = st.empty()

# Button Actions
if st.sidebar.button("Generate New Instance", key='generate_vrp'):
    route_cache = {}
    st.session_state.cities = (np.random.rand(num_customers, 2) * 100).tolist()
    st.session_state.demands = (np.random.rand(num_customers) * 0.9 + 0.1).tolist()
    st.session_state.depots = [(random.uniform(10, 90), random.uniform(10, 90)) for _ in range(num_depots)]
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_vrp_solution(st.session_state.cities, st.session_state.depots, [], "Generated Instance", ax, st.session_state.demands, params)
    plot_placeholder.pyplot(fig)

if st.sidebar.button("Run VNS Solver", key='solve_vns'):
    if 'cities' not in st.session_state:
        st.sidebar.error("Please generate an instance first.")
    else:
        route_cache = {}
        length_placeholder.text("")
        details_placeholder.text("")
        fig, ax = plt.subplots(figsize=(10, 7))
        start_time = time.time()
        best_solution, best_objective = vns_solver(
            st.session_state.cities, st.session_state.depots, st.session_state.demands, params,
            plot_placeholder, status_placeholder, length_placeholder, fig, ax)
        end_time = time.time()
        st.session_state.solution = best_solution
        st.session_state.objective = best_objective
        if best_solution is not None:
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
                    route_count += 1
            details_placeholder.text_area("Route Details", details_str, height=200)

# Display Initial/Last State
if 'cities' in st.session_state:
    fig, ax = plt.subplots(figsize=(10, 7))
    title = "Current VRP Instance"
    if 'solution' in st.session_state and st.session_state.solution is not None:
        title = f"Last Solution (Obj: {st.session_state.objective:.2f})"
        plot_vrp_solution(st.session_state.cities, st.session_state.depots, st.session_state.solution, title, ax, st.session_state.demands, params)
        length_placeholder.metric("Last Objective", f"{st.session_state.objective:.2f}")
    else:
        plot_vrp_solution(st.session_state.cities, st.session_state.depots, [], title, ax, st.session_state.demands, params)
    plot_placeholder.pyplot(fig)
else:
    status_placeholder.info("Click 'Generate New Instance' in the sidebar.")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_title("No Instance Generated")
    ax.set_xticks([])
    ax.set_yticks([])
    plot_placeholder.pyplot(fig)