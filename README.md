# Dragonfly Algorithm (DA) - Python Implementation

A complete Python implementation of the Dragonfly Algorithm based on the original MATLAB code by Seyedali Mirjalili.

## Overview

The Dragonfly Algorithm (DA) is a meta-heuristic optimization algorithm inspired by the swarming behavior of dragonflies. This implementation provides a faithful Python translation of the original MATLAB code with enhanced features and comprehensive benchmarking capabilities.

## Features

### Core Algorithm
- **5 Swarming Behaviors**: Separation, Alignment, Cohesion, Food attraction, Enemy distraction
- **Adaptive Parameters**: Weights that change over iterations for optimal performance
- **Levy Flight**: Exploration mechanism when no neighbors are found
- **Boundary Handling**: Proper handling of search space boundaries
- **Convergence Tracking**: Detailed convergence curve analysis

### Benchmark Functions
- **23 Test Functions**: Complete implementation of F1-F23 from the original paper
- **Multiple Dimensions**: Support for various problem dimensions (2D to 10D)
- **Different Characteristics**: Unimodal, multimodal, and complex optimization landscapes

### Visualization & Analysis
- **Convergence Plots**: Semi-logarithmic convergence curves
- **Function Comparison**: Side-by-side comparison of different functions
- **Parameter Studies**: Analysis of algorithm parameters' effects
- **Performance Metrics**: Execution time and convergence analysis

## Installation

1. Clone or download the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from dragonfly_algorithm import DragonflyAlgorithm
from benchmark_functions import BenchmarkFunctions

# Create DA instance
da = DragonflyAlgorithm(search_agents_no=40, max_iteration=500)

# Get function details
lb, ub, dim, fobj = BenchmarkFunctions.get_function_details('F1')

# Run optimization
best_score, best_pos, convergence_curve = da.optimize(fobj, dim, lb, ub)

print(f"Best Score: {best_score}")
print(f"Best Position: {best_pos}")
```

### Running the Demo

```bash
python demo.py
```

This will run a comprehensive demonstration including:
- Optimization on multiple benchmark functions
- Convergence analysis
- Function comparison
- Parameter studies

### Custom Objective Function

```python
def my_objective_function(x):
    """Your custom objective function"""
    return np.sum(x**2) + np.sin(x[0])

# Use with DA
da = DragonflyAlgorithm(40, 500)
best_score, best_pos, convergence_curve = da.optimize(
    my_objective_function, 
    dim=10, 
    lb=np.array([-100]*10), 
    ub=np.array([100]*10)
)
```

## Available Benchmark Functions

| Function | Name | Dimension | Bounds | Type |
|----------|------|-----------|--------|------|
| F1 | Sphere | 10 | [-100, 100] | Unimodal |
| F2 | Sum of Abs + Product | 10 | [-10, 10] | Unimodal |
| F3 | Sum of Squares | 10 | [-100, 100] | Unimodal |
| F4 | Maximum | 10 | [-100, 100] | Unimodal |
| F5 | Rosenbrock | 10 | [-30, 30] | Unimodal |
| F6 | Step | 10 | [-100, 100] | Unimodal |
| F7 | Quartic with Noise | 10 | [-1.28, 1.28] | Unimodal |
| F8 | Schwefel | 10 | [-500, 500] | Multimodal |
| F9 | Rastrigin | 10 | [-5.12, 5.12] | Multimodal |
| F10 | Ackley | 10 | [-32, 32] | Multimodal |
| F11 | Griewank | 10 | [-600, 600] | Multimodal |
| F12 | Penalized 1 | 10 | [-50, 50] | Multimodal |
| F13 | Penalized 2 | 10 | [-50, 50] | Multimodal |
| F14 | Shekel's Foxholes | 2 | [-65.536, 65.536] | Multimodal |
| F15 | Kowalik | 4 | [-5, 5] | Multimodal |
| F16 | Six-Hump Camel-Back | 2 | [-5, 5] | Multimodal |
| F17 | Branin | 2 | [-5, 0] × [10, 15] | Multimodal |
| F18 | Goldstein-Price | 2 | [-2, 2] | Multimodal |
| F19 | Hartman 3D | 3 | [0, 1] | Multimodal |
| F20 | Hartman 6D | 6 | [0, 1] | Multimodal |
| F21 | Shekel 5D | 4 | [0, 10] | Multimodal |
| F22 | Shekel 7D | 4 | [0, 10] | Multimodal |
| F23 | Shekel 10D | 4 | [0, 10] | Multimodal |

## Algorithm Parameters

- **search_agents_no**: Number of dragonflies (default: 40)
- **max_iteration**: Maximum number of iterations (default: 500)
- **Adaptive weights**: Separation (s), Alignment (a), Cohesion (c), Food attraction (f), Enemy distraction (e)
- **Inertia weight (w)**: Decreases from 0.9 to 0.4 over iterations
- **Control parameter (my_c)**: Decreases from 0.1 to 0 over first half of iterations

## Files Structure

```
├── dragonfly_algorithm.py    # Main DA implementation
├── benchmark_functions.py    # 23 benchmark functions
├── demo.py                  # Comprehensive demonstration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Performance

The Python implementation maintains the same mathematical formulation as the original MATLAB code while providing:

- **Faster execution** through NumPy vectorization
- **Better memory management** with efficient array operations
- **Enhanced visualization** with matplotlib
- **Comprehensive analysis tools** for research and comparison

## Reference

S. Mirjalili, "Dragonfly algorithm: a new meta-heuristic optimization technique for solving single-objective, discrete, and multi-objective problems," Neural Computing and Applications, DOI: http://dx.doi.org/10.1007/s00521-015-1920-1

## License

This implementation is based on the original MATLAB code by Seyedali Mirjalili. Please refer to the original license terms.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this implementation. 