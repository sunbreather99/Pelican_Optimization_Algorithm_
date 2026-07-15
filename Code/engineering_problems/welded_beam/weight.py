# Objective function: Minimize the fabrication cost
def weight(x):
    x1, x2, x3, x4 = x
    term1 = 1.10471 * x1**2 / x2
    term2 = 0.04811 * x3 * x4 * (14 + x2)
    return term1 + term2
