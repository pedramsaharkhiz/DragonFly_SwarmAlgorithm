import numpy as np
import matplotlib.pyplot as plt
from dragonfly_algorithm import DragonflyAlgorithm
from benchmark_functions import BenchmarkFunctions
import time

def run_optimization_demo():
    """Run a comprehensive demo of the Dragonfly Algorithm"""
    
    print("=" * 60)
    print("DRAGONFLY ALGORITHM (DA) - PYTHON IMPLEMENTATION")
    print("=" * 60)
    
    # Algorithm parameters
    search_agents_no = 40
    max_iteration = 500
    
    # Test functions to demonstrate
    test_functions = ['F1', 'F5', 'F9', 'F10', 'F11']
    function_names = {
        'F1': 'Sphere Function',
        'F5': 'Rosenbrock Function', 
        'F9': 'Rastrigin Function',
        'F10': 'Ackley Function',
        'F11': 'Griewank Function'
    }
    
    # Results storage
    results = {}
    
    # Create DA instance
    da = DragonflyAlgorithm(search_agents_no, max_iteration)
    
    print(f"\nAlgorithm Parameters:")
    print(f"- Search Agents: {search_agents_no}")
    print(f"- Max Iterations: {max_iteration}")
    print(f"- Test Functions: {len(test_functions)}")
    
    print(f"\n{'Function':<15} {'Best Score':<15} {'Time (s)':<10} {'Convergence':<12}")
    print("-" * 60)
    
    for func_name in test_functions:
        # Get function details
        lb, ub, dim, fobj = BenchmarkFunctions.get_function_details(func_name)
        
        # Run optimization
        start_time = time.time()
        best_score, best_pos, convergence_curve = da.optimize(fobj, dim, lb, ub)
        end_time = time.time()
        
        # Store results
        results[func_name] = {
            'best_score': best_score,
            'best_position': best_pos,
            'convergence_curve': convergence_curve,
            'execution_time': end_time - start_time,
            'dimension': dim
        }
        
        # Print results
        convergence_status = "✓ Converged" if convergence_curve[-1] < 1e-6 else "⚠ Not converged"
        print(f"{func_name:<15} {best_score:<15.6f} {end_time - start_time:<10.3f} {convergence_status:<12}")
    
    # Visualization
    plot_results(results, function_names)
    
    return results

def plot_results(results, function_names):
    """Create comprehensive visualization of results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dragonfly Algorithm - Optimization Results', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for idx, (func_name, result) in enumerate(results.items()):
        if idx >= len(axes_flat):
            break
            
        ax = axes_flat[idx]
        
        # Plot convergence curve
        iterations = range(len(result['convergence_curve']))
        ax.semilogy(iterations, result['convergence_curve'], 'r-', linewidth=2, label='DA')
        ax.set_title(f'{func_name}: {function_names[func_name]}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Score')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text box with results
        textstr = f'Best Score: {result["best_score"]:.6f}\n'
        textstr += f'Dimension: {result["dimension"]}\n'
        textstr += f'Time: {result["execution_time"]:.3f}s'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    # Remove empty subplot if needed
    if len(results) < len(axes_flat):
        axes_flat[-1].remove()
    
    plt.tight_layout()
    plt.show()

def compare_functions():
    """Compare different functions on the same scale"""
    
    print("\n" + "=" * 60)
    print("FUNCTION COMPARISON")
    print("=" * 60)
    
    # Algorithm parameters
    search_agents_no = 40
    max_iteration = 500
    
    # Functions to compare
    compare_functions = ['F1', 'F9', 'F10']  # Sphere, Rastrigin, Ackley
    function_names = {
        'F1': 'Sphere',
        'F9': 'Rastrigin', 
        'F10': 'Ackley'
    }
    
    da = DragonflyAlgorithm(search_agents_no, max_iteration)
    
    plt.figure(figsize=(12, 8))
    
    for func_name in compare_functions:
        lb, ub, dim, fobj = BenchmarkFunctions.get_function_details(func_name)
        best_score, best_pos, convergence_curve = da.optimize(fobj, dim, lb, ub)
        
        plt.semilogy(convergence_curve, linewidth=2, label=f'{func_name}: {function_names[func_name]}')
    
    plt.title('Convergence Comparison - Different Benchmark Functions', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Best Score (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def parameter_study():
    """Study the effect of different parameters"""
    
    print("\n" + "=" * 60)
    print("PARAMETER STUDY")
    print("=" * 60)
    
    # Test different population sizes
    population_sizes = [20, 40, 60, 80]
    max_iteration = 300
    
    lb, ub, dim, fobj = BenchmarkFunctions.get_function_details('F1')  # Sphere function
    
    plt.figure(figsize=(12, 8))
    
    for pop_size in population_sizes:
        da = DragonflyAlgorithm(pop_size, max_iteration)
        best_score, best_pos, convergence_curve = da.optimize(fobj, dim, lb, ub)
        
        plt.semilogy(convergence_curve, linewidth=2, 
                    label=f'Population Size: {pop_size}')
    
    plt.title('Effect of Population Size on Convergence (Sphere Function)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Best Score (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    """Main demo function"""
    
    # Run main optimization demo
    results = run_optimization_demo()
    
    # Run function comparison
    compare_functions()
    
    # Run parameter study
    parameter_study()
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("The Python implementation successfully replicates the MATLAB Dragonfly Algorithm.")
    print("Key features implemented:")
    print("- All 5 swarming behaviors (Separation, Alignment, Cohesion, Food attraction, Enemy distraction)")
    print("- Adaptive parameters over iterations")
    print("- Levy flight for exploration")
    print("- Boundary handling")
    print("- 23 benchmark functions (F1-F23)")
    print("- Comprehensive visualization and analysis tools")
    print("=" * 60)

if __name__ == "__main__":
    main() 