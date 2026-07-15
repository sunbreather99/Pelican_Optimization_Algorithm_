import numpy as np

def POA(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    best_so_far = []  # Store best fitness per iteration
    fbest = float('inf')

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

    # Initialize alpha, beta, delta wolves
    alpha_position = np.zeros(dimension)
    alpha_fitness = float('inf')

    beta_position = np.zeros(dimension)
    beta_fitness = float('inf')

    delta_position = np.zeros(dimension)
    delta_fitness = float('inf')

    for t in range(1, Max_iterations + 1):
        best = np.min(fit)
        location = np.argmin(fit)

        if (t == 1) or (best < fbest):
            Xbest = X[location, :].copy()
            fbest = best

        # Random food location (best solution found so far)
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

            # Bounds check
            X_new = np.clip(X_new, lowerbound, upperbound)

            # Selection
            f_new = fitness(X_new)
            if f_new <= fit[i]:
                X[i, :] = X_new
                fit[i] = f_new

            # ----- GWO Phase (Grey Wolf Optimizer) -----
            # Update the Alpha, Beta, and Delta wolves based on fitness
            sorted_fits = np.argsort(fit)
            alpha_position = X[sorted_fits[0], :]
            alpha_fitness = fit[sorted_fits[0]]

            if len(sorted_fits) > 1:
                beta_position = X[sorted_fits[1], :]
                beta_fitness = fit[sorted_fits[1]]

            if len(sorted_fits) > 2:
                delta_position = X[sorted_fits[2], :]
                delta_fitness = fit[sorted_fits[2]]

            # Update agent positions based on GWO equation
            a = 2 - 2 * t / Max_iterations  # Linearly decreasing factor
            r1 = np.random.rand()
            r2 = np.random.rand()

            # Alpha, Beta, and Delta attraction
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * alpha_position - X[i, :])
            X1 = alpha_position - A1 * D_alpha

            r1 = np.random.rand()
            r2 = np.random.rand()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * beta_position - X[i, :])
            X2 = beta_position - A2 * D_beta

            r1 = np.random.rand()
            r2 = np.random.rand()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * delta_position - X[i, :])
            X3 = delta_position - A3 * D_delta

            # Update position of the agent
            X_new = (X1 + X2 + X3) / 3
            X_new = np.clip(X_new, lowerbound, upperbound)

            # Evaluate fitness of new position
            f_new = fitness(X_new)
            if f_new < fit[i]:
                X[i, :] = X_new
                fit[i] = f_new

        # Store best fitness found
        best_so_far.append(fbest)

    return fbest, Xbest, best_so_far
