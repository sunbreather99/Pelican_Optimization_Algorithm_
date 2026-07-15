import matplotlib.pyplot as plt

def plot_func(Max_iterations, avg_curve, func_name):
    plt.figure()
    plt.plot(range(Max_iterations), avg_curve, label='Average Convergence')
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title(f"Convergence Curve for {func_name}")
    plt.legend()
    plt.grid(True)

    # Save the plot instead of showing it
    plt.savefig(f"POA_results/{func_name}_convergence.png")
    plt.close()  # Close the figure to avoid blocking