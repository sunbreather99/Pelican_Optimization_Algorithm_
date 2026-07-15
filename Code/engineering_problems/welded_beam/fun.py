import getconstraints as g
import constraints
import weight

# Fitness function that adds penalty for constraint violations
def Fun(fhandle, fnonlin, u):
    z = fhandle(u)
    # Apply nonlinear constraints using penalty method
    z += g.getconstraints(fnonlin, u)
    return z
