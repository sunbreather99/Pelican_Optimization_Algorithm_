import POA
import DE_POA
import SA_POA
import GA_POA
import ACO_POA
import GWO_POA
import plotter as p
import cec2022 as c
import numpy as np
import os

# Configuration
SearchAgents = 50
Max_iterations = 1200
Num_trials = 3

# Create output directory if needed
output_dir = "POA_results"
os.makedirs(output_dir, exist_ok=True)

# Loop through all 30 benchmark functions
for func_index in range(22, 25):
    Fun_name = f"F{func_index}"
    print(f"\nRunning POA on {Fun_name}...")

    best_scores = []
    all_curves = []

    for i in range(Num_trials):
        print("trial", i+1)
        fitness, lowerbound, upperbound, dimension = c.Get_F(Fun_name)
        Best_score, Best_pos, POA_curve = POA.POA(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness)
        best_scores.append(Best_score)
        all_curves.append(POA_curve)

    # Print summary statistics
    best_scores_np = np.array(best_scores)
    print(f"Best Score: {np.min(best_scores_np)}")
    print(f"Worst Score: {np.max(best_scores_np)}")
    print(f"Mean Score: {np.mean(best_scores_np)}")
    print(f"Std Deviation: {np.std(best_scores_np)}")

    # Save results to file
    result_file = os.path.join(output_dir, f"{Fun_name}_results.txt")
    with open(result_file, "w") as f_out:
        f_out.write(f"Function: {Fun_name}\n")
        f_out.write(f"Best Score: {np.min(best_scores_np)}\n")
        f_out.write(f"Worst Score: {np.max(best_scores_np)}\n")
        f_out.write(f"Mean Score: {np.mean(best_scores_np)}\n")
        f_out.write(f"Standard Deviation: {np.std(best_scores_np)}\n")
        f_out.write("All Best Scores:\n")
        f_out.write(", ".join(map(str, best_scores)) + "\n")

    # Plot average convergence curve
    average_curve = np.mean(all_curves, axis=0)
    p.plot_func(Max_iterations, average_curve, Fun_name)