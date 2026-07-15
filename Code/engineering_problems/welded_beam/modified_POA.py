import numpy as np

def DE_mutation(target_idx, X, F, bounds):
    """DE/rand/1 mutation strategy."""
    idxs = [idx for idx in range(len(X)) if idx != target_idx]
    a, b, c = X[np.random.choice(idxs, 3, replace=False)]
    mutant = a + F * (b - c)
    mutant = np.clip(mutant, bounds[0], bounds[1])
    return mutant

def DE_crossover(target, mutant, CR):
    """Binomial crossover."""
    d = len(target)
    j_rand = np.random.randint(d)
    trial = np.array([mutant[j] if (np.random.rand() < CR or j == j_rand) else target[j] for j in range(d)])
    return trial

def POA(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness, fhandle, fnonlin,
                       F_de=0.5, CR=0.9):
    best_so_far = []
    fbest = np.inf

    lowerbound = np.ones(dimension) * lowerbound
    upperbound = np.ones(dimension) * upperbound
    bounds = (lowerbound, upperbound)

    # Initialization
    X = lowerbound + np.random.rand(SearchAgents, dimension) * (upperbound - lowerbound)
    fit = np.array([fitness(fhandle, fnonlin, X[i, :]) for i in range(SearchAgents)])

    for t in range(1, Max_iterations + 1):
        best = np.min(fit)
        location = np.argmin(fit)
        if best < fbest:
            fbest = best
            Xbest = X[location, :]

        k = np.random.permutation(SearchAgents)[0]
        X_FOOD = X[k, :]
        F_FOOD = fit[k]

        for i in range(SearchAgents):
            # === POA Exploration ===
            I = 1 + np.random.randint(2)
            if fit[i] > F_FOOD:
                X_new = X[i, :] + np.random.rand() * (X_FOOD - I * X[i, :])
            else:
                X_new = X[i, :] + np.random.rand() * (X[i, :] - X_FOOD)

            X_new = np.clip(X_new, lowerbound, upperbound)
            f_new = fitness(fhandle, fnonlin, X_new)
            if f_new <= fit[i]:
                X[i, :] = X_new
                fit[i] = f_new

            # === DE Phase ===
            mutant = DE_mutation(i, X, F_de, bounds)
            trial = DE_crossover(X[i, :], mutant, CR)
            trial = np.clip(trial, lowerbound, upperbound)
            f_trial = fitness(fhandle, fnonlin, trial)
            if f_trial < fit[i]:
                X[i, :] = trial
                fit[i] = f_trial

            # === POA Exploitation ===
            r = np.random.rand()
            X_new = X[i, :] + (0.2 * (1 - t / Max_iterations) * (2 * r - 1) * X[i, :])
            X_new = np.clip(X_new, lowerbound, upperbound)
            f_new = fitness(fhandle, fnonlin, X_new)
            if f_new <= fit[i]:
                X[i, :] = X_new
                fit[i] = f_new

        best_so_far.append(fbest)

    return fbest, Xbest, best_so_far
