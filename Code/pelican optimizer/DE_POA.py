import numpy as np


def POA(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    
    best_so_far = []  # Store best fitness per iteration
    fbest = 0

    # Bound vectors
    lowerbound = np.ones(dimension) * lowerbound
    upperbound = np.ones(dimension) * upperbound

    # Initialize agent positions
    X = np.zeros((SearchAgents, dimension))
    for i in range(dimension):
        X[:, i] = lowerbound[i] + np.random.rand(SearchAgents) * (upperbound[i] - lowerbound[i])

    # Evaluate initial fitness
    fit = np.zeros(SearchAgents)
    for i in range(SearchAgents):
        fit[i] = fitness(X[i, :])
    
    for t in range(1, Max_iterations + 1):
        best = np.min(fit)
        location = np.argmin(fit)
        
        if (t == 1) or (best < fbest):
            Xbest = X[location, :].copy()
            fbest = best

        # Random food location
        k = np.random.permutation(SearchAgents)[0]
        X_FOOD = X[k, :].copy()
        F_FOOD = fit[k]

        # DE Parameters
        F_de = 0.5  # Differential weight
        CR = 0.9    # Crossover rate
        DE_RATE = 0.9

        for i in range(SearchAgents):
            
            # ----- Differential Evolution Phase (Exploration Enhancement) -----
            if np.random.rand() < DE_RATE:  # Apply DE 
                idxs = [idx for idx in range(SearchAgents) if idx != i]
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

                # Mutation
                V = X[r1] + F_de * (X[r2] - X[r3])

                # Crossover
                jrand = np.random.randint(dimension)
                U = np.array([
                    V[j] if (np.random.rand() < CR or j == jrand) else X[i][j]
                    for j in range(dimension)
                ])

                # Bounds check
                U = np.clip(U, lowerbound, upperbound)

                # Selection
                f_U = fitness(U)
                if f_U < fit[i]:
                    X[i, :] = U
                    fit[i] = f_U

            # ----- Phase 1: Exploration (towards prey) -----
            I = 1 + np.random.randint(2)
            if fit[i] > F_FOOD:
                X_new = X[i, :] + np.random.rand() * (X_FOOD - I * X[i, :])
            else:
                X_new = X[i, :] + np.random.rand() * (X[i, :] - X_FOOD)

            # Bound check
            X_new = np.clip(X_new, lowerbound, upperbound)

            # Selection
            f_new = fitness(X_new)
            if f_new <= fit[i]:
                X[i, :] = X_new
                fit[i] = f_new

            # ----- Phase 2: Exploitation (winging) -----
            r = np.random.rand()
            X_new = X[i, :] + (0.2 * (1 - t / Max_iterations) * (2 * r - 1) * X[i, :])

            X_new = np.clip(X_new, lowerbound, upperbound)

            f_new = fitness(X_new)
            if f_new <= fit[i]:
                X[i, :] = X_new
                fit[i] = f_new

        best_so_far.append(fbest)

    return fbest, Xbest, best_so_far

