import numpy as np

# CEC 2022 Benchmark Functions

def Get_F(F):
    if F == 'F1':
        F_obj = F1
    elif F == 'F2':
        F_obj = F2
    elif F == 'F3':
        F_obj = F3
    elif F == 'F4':
        F_obj = F4
    elif F == 'F5':
        F_obj = F5
    elif F == 'F6':
        F_obj = F6
    elif F == 'F7':
        F_obj = F7
    elif F == 'F8':
        F_obj = F8
    elif F == 'F9':
        F_obj = F9
    elif F == 'F10':
        F_obj = F10
    elif F == 'F11':
        F_obj = F11
    elif F == 'F12':
        F_obj = F12
    elif F == 'F13':
        F_obj = F13
    elif F == 'F14':
        F_obj = F14
    elif F == 'F15':
        F_obj = F15
    elif F == 'F16':
        F_obj = F16
    elif F == 'F17':
        F_obj = F17
    elif F == 'F18':
        F_obj = F18
    elif F == 'F19':
        F_obj = F19
    elif F == 'F20':
        F_obj = F20
    elif F == 'F21':
        F_obj = F21
    elif F == 'F22':
        F_obj = F22
    elif F == 'F23':
        F_obj = F23
    elif F == 'F24':
        F_obj = F24
    else:
        raise ValueError(f"Function {F} is not defined in CEC 2022.")

    # Standard domain and dimensionality
    LB = -100
    UB = 100
    Dim = 30

    return F_obj, LB, UB, Dim


def F1(x):
    """Shifted and Rotated Sphere Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(rotated_x**2)

def F2(x):
    """Shifted and Rotated Schwefel Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return 418.9829 * len(x) - np.sum(rotated_x * np.sin(np.sqrt(np.abs(rotated_x))))

def F3(x):
    """Shifted and Rotated Rastrigin Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return 10 * len(x) + np.sum(rotated_x**2 - 10 * np.cos(2 * np.pi * rotated_x))

def F4(x):
    """Shifted and Rotated Ackley Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    part1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(rotated_x**2)))
    part2 = -np.exp(0.5 * np.sum(np.cos(2 * np.pi * rotated_x)))
    return part1 + part2 + np.e + 20

def F5(x):
    """Shifted and Rotated Griewangk Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return 1 + np.sum(rotated_x**2) / 4000 - np.prod(np.cos(rotated_x / np.sqrt(np.arange(1, len(rotated_x) + 1))))

def F6(x):
    """Shifted and Rotated Rosenbrock Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(100 * (rotated_x[1:] - rotated_x[:-1]**2)**2 + (1 - rotated_x[:-1])**2)

def F7(x):
    """Shifted and Rotated Michalewicz Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    m = 10
    return -np.sum(np.sin(rotated_x) * np.sin(np.arange(1, len(rotated_x) + 1) * rotated_x**2 / np.pi)**(2 * m))

def F8(x):
    """Shifted and Rotated Weierstrass Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    a = 0.5
    b = 3
    return np.sum(np.abs(np.sin(np.pi * rotated_x) + 0.5 * np.cos(np.pi * rotated_x)))

def F9(x):
    """Shifted and Rotated Happy Cat Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(rotated_x**2) - 10 * np.cos(2 * np.pi * rotated_x) + 10

def F10(x):
    """Shifted and Rotated Hybrid Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(np.abs(np.sin(rotated_x)) + np.cos(rotated_x)**2)

def F11(x):
    """Shifted and Rotated Cigar Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return rotated_x[0]**2 + 10 * np.sum(rotated_x[1:]**2)

def F12(x):
    """Shifted and Rotated Elliptic Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(rotated_x**2 / (10**np.arange(len(x))))

def F13(x):
    """Shifted and Rotated Hybrid Function 2"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(np.abs(np.sin(rotated_x)) + np.cos(rotated_x)**2)

def F14(x):
    """Shifted and Rotated Step Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(np.floor(rotated_x))

def F15(x):
    """Shifted and Rotated Ackley Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    part1 = -20 * np.exp(-0.2 * np.sqrt(0.5 * np.sum(rotated_x**2)))
    part2 = -np.exp(0.5 * np.sum(np.cos(2 * np.pi * rotated_x)))
    return part1 + part2 + np.e + 20

def F16(x):
    """Shifted and Rotated Griewangk Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return 1 + np.sum(rotated_x**2) / 4000 - np.prod(np.cos(rotated_x / np.sqrt(np.arange(1, len(rotated_x) + 1))))

def F17(x):
    """Shifted and Rotated Rosenbrock Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(100 * (rotated_x[1:] - rotated_x[:-1]**2)**2 + (1 - rotated_x[:-1])**2)

def F18(x):
    """Shifted and Rotated Schwefel Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return 418.9829 * len(x) - np.sum(rotated_x * np.sin(np.sqrt(np.abs(rotated_x))))

def F19(x):
    """Shifted and Rotated Michalewicz Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    m = 10
    return -np.sum(np.sin(rotated_x) * np.sin(np.arange(1, len(rotated_x) + 1) * rotated_x**2 / np.pi)**(2 * m))

def F20(x):
    """Shifted and Rotated Step Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(np.floor(rotated_x))

def F21(x):
    """Shifted and Rotated Happy Cat Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(rotated_x**2) - 10 * np.cos(2 * np.pi * rotated_x) + 10

def F22(x):
    """Shifted and Rotated Elliptic Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(rotated_x**2 / (10**np.arange(len(x))))

def F23(x):
    """Shifted and Rotated Hybrid Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(np.abs(np.sin(rotated_x)) + np.cos(rotated_x)**2)

def F24(x):
    """Shifted and Rotated Step Function"""
    shift = np.random.uniform(-100, 100, len(x))  # Shift values
    A = np.random.rand(len(x), len(x))  # Random rotation matrix
    rotated_x = np.dot(x - shift, A)  # Apply shift and rotation
    return np.sum(np.floor(rotated_x))
