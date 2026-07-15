import numpy as np

# Constraint functions
def constraint(x):
    x1, x2, x3, x4 = x

    # τ(x)
    tau_0 = 6000 / np.sqrt(2 * x1 * x2)
    tau_00 = 6000 * (14 + x2) / (x1 * x2)
    R = np.sqrt(x2**2 / 4 + (x1 + x3)**2)
    J = (x1 * x2 / np.sqrt(2)) * (x2**2 / 12 + (x1 + x3)**2)

    tau_x = np.sqrt(tau_0 + (2 * tau_0 * x2**2 / (2 * R)) + (tau_00 / 2))

    # σ(x)
    sigma_x = 504000 / (x4 * x3**2)

    # δ(x)
    delta_x = 65856000 / (30 * 10**6 * x4 * x3**3)

    # pc(x)
    pc_x = 4.013

    # Define inequality constraints
    g1 = tau_x - 13600  # Constraint for τ(x)
    g2 = sigma_x - 30000  # Constraint for σ(x)
    g3 = x1 - x4  # Geometric constraint
    g4 = 0.10471 * x1**2 + 0.04811 * x3 * x4 * (14 + x2) - 5.0  # Fabrication cost constraint
    g5 = 0.125 - x1  # Geometric limit constraint
    g6 = delta_x - 0.25  # Displacement constraint
    g7 = 6000 - pc_x  # Power constraint

    return np.array([g1, g2, g3, g4, g5, g6, g7])
