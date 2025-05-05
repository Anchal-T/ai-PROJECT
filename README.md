# Emergency Aerial Vehicle Scheduling via Graph Neural Neighborhood Search

This repository contains an implementation of the research paper "Emergency Scheduling of Aerial Vehicles via Graph Neural Neighborhood Search" (IEEE Transactions on Artificial Intelligence, 2025).

🔗 **Live Demo**: [https://neighbourhoodsearch.streamlit.app/](https://neighbourhoodsearch.streamlit.app/)

## Overview

This project implements an innovative approach to emergency aerial vehicle scheduling using Graph Neural Networks (GNN) and neighborhood search techniques. The implementation provides both a visualization tool and a practical solver for Vehicle Routing Problems (VRP) with a focus on emergency scenarios.

## Features

- **Graph Neural Neighborhood Search**
  - Interactive visualization of VRP instances
  - K-nearest neighbor computation
  - Greedy algorithm implementation
  - 2-opt local search improvement
  - Demand-aware route planning

- **Aerial Task Scheduling**
  - GNN-based task assignment optimization
  - Vehicle-task compatibility scoring
  - Feature-rich scheduling visualization
  - Comparative analysis with naive approaches
  - Real-time solution improvement

## Technical Components

### Core Modules

1. **GraphNeighborhoodSearch** (`graph_neighborhood_search.py`)
   - Random problem generation
   - Nearest neighbor computation
   - Greedy VRP solver
   - Route visualization
   - Local search optimization

2. **AerialScheduling** (`aerial_scheduling.py`)
   - Vehicle and task feature management
   - Graph construction
   - GNN model implementation
   - Task assignment optimization

3. **Web Interface** (`app.py`)
   - Streamlit-based interactive UI
   - Real-time visualization
   - Parameter tuning
   - Solution comparison

### Models

- **Enhanced GNN Architecture**
  - Graph Convolutional layers
  - Attention mechanisms
  - Multi-head attention fusion
  - Dropout regularization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aerial-scheduling-gnn.git
cd aerial-scheduling-gnn

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Web Application

```bash
streamlit run app.py
```

### Using the API

```python
from graph_neighborhood_search import GraphNeighborhoodSearch
from aerial_scheduling import AerialScheduling

# Create a VRP instance
solver = GraphNeighborhoodSearch(n_nodes=20, k=5)
solver.generate_random_problem()
solver.greedy_vrp_solve()

# Create an aerial scheduling instance
scheduler = AerialScheduling(num_vehicles=5, num_tasks=10)
graph_data = scheduler.get_graph_data()
```

## Implementation Details

### Vehicle Features
- Fuel capacity (50-100%)
- Speed (100-200 km/h)
- Payload capacity (5-20 units)

### Task Features
- Urgency (1-10)
- Complexity (1-10)
- Required payload (1-15 units)

### Optimization Techniques
1. **Greedy Algorithm**
   - Initial solution construction
   - Capacity constraints
   - Distance minimization

2. **2-opt Local Search**
   - Route improvement
   - Edge swapping
   - Local optimality

3. **GNN-based Assignment**
   - Node embedding
   - Edge weight consideration
   - Compatibility scoring

## Results

The implementation demonstrates significant improvements over naive approaches:
- Reduced total route distances
- Better task-vehicle matching
- Improved emergency response planning
- Real-time solution generation

## Citation

```bibtex
@article{guo2025emergency,
  title={Emergency Scheduling of Aerial Vehicles via Graph Neural Neighborhood Search},
  author={Guo, T. and Mei, Y. and Du, W. and Lv, Y. and Li, Y. and Song, T.},
  journal={IEEE Transactions on Artificial Intelligence},
  volume={1},
  number={1},
  pages={1-20},
  year={2025},
  doi={10.1109/TAI.2025.3528381}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributors

- [Your Name]
- [Other Contributors]

## Acknowledgments

This implementation is based on the research paper by T. Guo et al. (2025). Special thanks to the authors for their innovative approach to emergency aerial vehicle scheduling.

## References

T. Guo, Y. Mei, W. Du, Y. Lv, Y. Li and T. Song, "Emergency Scheduling of Aerial Vehicles via Graph Neural Neighborhood Search," in IEEE Transactions on Artificial Intelligence, vol. 1, no. 1, pp. 1-20, 2025, doi: 10.1109/TAI.2025.3528381.