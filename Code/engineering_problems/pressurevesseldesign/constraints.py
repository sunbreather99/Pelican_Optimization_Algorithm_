import numpy as np

def constraint(x):
    # Round x1 and x2 to nearest multiple of 0.0625 (discrete variables)
    x1 = round(x[0] / 0.0625) * 0.0625
    x2 = round(x[1] / 0.0625) * 0.0625
    x3 = x[2]
    x4 = x[3]

    # Constraint 1: -x1 + 0.0193 * x3 ≤ 0
    g1 = -x1 + 0.0193 * x3

    # Constraint 2: -x2 + 0.00954 * x3 ≤ 0
    g2 = -x2 + 0.00954 * x3

    # Constraint 3: Volume constraint: volume must be ≥ 1296000
    g3 = -np.pi * x3**2 * x4 - (4/3) * np.pi * x3**3 + 1296000

    # Constraint 4: x4 ≤ 240
    g4 = x4 - 240

    # Custom constraints to penalize:
    # x1, x2 ≥ 0.0625
    g5 = 0.0625 - x1
    g6 = 0.0625 - x2

    # x3, x4 ≥ 10
    g7 = 10 - x3
    g8 = 10 - x4

    return np.array([g1, g2, g3, g4, g5, g6, g7, g8])
