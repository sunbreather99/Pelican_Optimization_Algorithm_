import numpy as np
import POA
import modified_POA
import weight
import constraints
import fun
import plotter
import writer

ROWS = 1
# --------------------------------------------------------------------------------------------------------------------------------------------------
SearchAgents = 50  # number of agents (population members)
Max_iterations = 1200  # maximum number of iterations (1_000)

# Object function information
lowerbound = [0.05, 0.1, 0.01, 0.05]  # Lower bounds for [h, l, t, b]
upperbound = [2.0, 10.0, 0.5, 1.0]  # Upper bounds for [h, l, t, b]
dimension = 4  # Number of variables
fitness = fun.Fun
fhandle = weight.weight
fnonlin = constraints.constraint

for i in range(1, ROWS + 1):
    # Calculating the solution of the given problem using POA    
    Best_score, Best_pos, POA_curve = modified_POA.POA(SearchAgents,
        Max_iterations, lowerbound, upperbound, dimension,
        fitness, fhandle, fnonlin)

    # Displaying results
    print(f"{i}) Best solution: {Best_pos}")
    print(f"Best Score: {Best_score}\n")
    plotter.plot_func(Max_iterations, POA_curve, 'Objective Space')
    
    # Save scores in a text file
    #writer.add_score(Best_pos, Best_score)
# --------------------------------------------------------------------------------------------------------------------------------------------------
