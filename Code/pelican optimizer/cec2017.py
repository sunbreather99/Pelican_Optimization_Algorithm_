import numpy as np

# Simulate loading shift vectors (you'll replace this with actual load function later)
def get_shift_vector(dim, seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(-100, 100, size=dim)

# Set up dimensions and shift
def shifted(x, shift_vector):
    return x - shift_vector

# F1: Shifted Sphere
def F1(x):
    o = get_shift_vector(len(x), seed=1)
    z = shifted(x, o)
    return np.sum(z**2)

# F2: Shifted Schwefel 1.2
def F2(x):
    o = get_shift_vector(len(x), seed=2)
    z = shifted(x, o)
    return np.sum([np.sum(z[:i+1])**2 for i in range(len(z))])

# F3: Shifted Rosenbrock
def F3(x):
    o = get_shift_vector(len(x), seed=3)
    z = shifted(x, o) + 1  # Important! Rosenbrock is usually centered at (1,...,1)
    return np.sum(100 * (z[1:] - z[:-1]**2)**2 + (z[:-1] - 1)**2)

# F4: Shifted Rastrigin
def F4(x):
    o = get_shift_vector(len(x), seed=4)
    z = shifted(x, o)
    return np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)

# F5: Shifted Griewank
def F5(x):
    o = get_shift_vector(len(x), seed=5)
    z = shifted(x, o)
    return 1 + np.sum(z**2) / 4000 - np.prod(np.cos(z / np.sqrt(np.arange(1, len(z)+1))))

# F6: Shifted Ackley
def F6(x):
    o = get_shift_vector(len(x), seed=6)
    z = shifted(x, o)
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(z**2))) - np.exp(np.mean(np.cos(2 * np.pi * z))) + 20 + np.e

# F7: Shifted Weierstrass (fixed)
def F7(x):
    o = get_shift_vector(len(x), seed=7)
    z = shifted(x, o)
    a, b, k_max = 0.5, 3, 20
    k = np.arange(k_max).reshape(-1, 1)
    term1 = np.sum(a**k * np.cos(2 * np.pi * b**k * z), axis=0).sum()
    term2 = len(x) * np.sum(a**k[:, 0] * np.cos(2 * np.pi * b**k[:, 0]))
    return term1 - term2

# F8: Shifted Non-continuous Rastrigin
def F8(x):
    o = get_shift_vector(len(x), seed=8)
    z = shifted(x, o)
    z[np.abs(z) < 0.5] = z[np.abs(z) < 0.5]
    z[np.abs(z) >= 0.5] = np.round(2 * z[np.abs(z) >= 0.5]) / 2
    return np.sum(z**2 - 10 * np.cos(2 * np.pi * z) + 10)

# F9: Shifted Schwefel
def F9(x):
    o = get_shift_vector(len(x), seed=9)
    z = shifted(x, o)
    return 418.9829 * len(x) - np.sum(z * np.sin(np.sqrt(np.abs(z))))

# F10: Shifted Elliptic
def F10(x):
    o = get_shift_vector(len(x), seed=10)
    z = shifted(x, o)
    return np.sum([10**6**(i / (len(z)-1)) * z[i]**2 for i in range(len(z))])

def F11(x):
    o = get_shift_vector(len(x), seed=11)
    z = shifted(x, o)
    return np.sum(z**4) - 16 * np.sum(z**2) + 5 * np.sum(z)

def F12(x):
    o = get_shift_vector(len(x), seed=12)
    z = shifted(x, o)
    return np.sum(0.5 * (z[:-1]**2 + z[1:]**2)) + np.prod(np.cos(z))

def F13(x):
    o = get_shift_vector(len(x), seed=13)
    z = shifted(x, o)
    return np.sum(np.abs(z)**(2 + 4 * np.arange(len(z)) / (len(z)-1)))

def F14(x):
    o = get_shift_vector(len(x), seed=14)
    z = shifted(x, o)
    term1 = np.sum(z**2)
    term2 = np.sum(0.5 * np.arange(1, len(z) + 1) * z)
    return term1 + term2**2 + term2**4

def F15(x):
    o = get_shift_vector(len(x), seed=15)
    z = shifted(x, o)
    return np.sin(np.pi * z[0])**2 + np.sum((z[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * z[:-1] + 1)**2)) + (z[-1] - 1)**2 * (1 + np.sin(2 * np.pi * z[-1])**2)

def F16(x):
    o = get_shift_vector(len(x), seed=16)
    z = shifted(x, o)
    m = 10
    return -np.sum(np.sin(z) * (np.sin((np.arange(len(z)) + 1) * z**2 / np.pi))**(2 * m))

def F17(x):
    o = get_shift_vector(len(x), seed=17)
    z = shifted(x, o)
    return np.sum(np.abs(z)**(2 + 4 * np.arange(len(z)) / (len(z)-1)))

def F18(x):
    return F4(x)  # Shifted Rastrigin

def F19(x):
    return F6(x)  # Shifted Ackley

def F20(x):
    return F7(x)  # Shifted Weierstrass

def F21(x):
    return (F1(x) + F4(x) + F5(x)) / 3

def F22(x):
    return (F6(x) + F9(x) + F10(x)) / 3

def F23(x):
    return (F14(x) + F15(x) + F16(x)) / 3

def F24(x):
    return (F3(x) + F12(x)) / 2

def F25(x):
    return (F6(x) + F1(x) + F7(x)) / 3

def F26(x):
    return (F5(x) + F4(x) + F15(x)) / 3

def F27(x):
    return (F14(x) + F3(x) + F4(x)) / 3

def F28(x):
    return (F1(x) + F10(x) + F12(x)) / 3

def F29(x):
    return (F13(x) + F4(x) + F6(x)) / 3

def F30(x):
    return (F9(x) + F5(x) + F7(x)) / 3

# Map and Get_F
function_map = {
    f"F{i}": func for i, func in enumerate([
        F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
        F11, F12, F13, F14, F15, F16, F17, F18, F19, F20,
        F21, F22, F23, F24, F25, F26, F27, F28, F29, F30
    ], start=1)
}

def Get_F(F_name):
    dim = 30
    lb = -100
    ub = 100
    if F_name in function_map:
        return function_map[F_name], lb, ub, dim
    else:
        raise Exception(f"Function {F_name} not defined.")