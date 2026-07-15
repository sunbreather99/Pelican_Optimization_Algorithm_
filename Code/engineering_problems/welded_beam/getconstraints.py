def getconstraints(fnonlin, u):
    # Penalty constant (large value)
    R = 10**15
    z = 0
    
    # Get nonlinear constraints
    g = fnonlin(u)
    
    # Apply all inequality constraints
    for gx in g:
        z += R * (getH(gx) ** 2)  # Penalty for violated constraints
    return z

def getH(g):
    # H(x) is used to apply penalty to violated constraints
    return max(0, g)
