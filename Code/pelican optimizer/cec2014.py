# cec2014_get_F.py

from opfunu.cec_based.cec2014 import (
    F12014, F22014, F32014, F42014, F52014, F62014, F72014, F82014, F92014, F102014,
    F112014, F122014, F132014, F142014, F152014, F162014, F172014, F182014, F192014, F202014,
    F212014, F222014, F232014, F242014, F252014, F262014, F272014, F282014, F292014, F302014
)

# Mapping function names to classes
cec_functions = {
    f"F{i}": func for i, func in enumerate([
        F12014, F22014, F32014, F42014, F52014, F62014, F72014, F82014, F92014, F102014,
        F112014, F122014, F132014, F142014, F152014, F162014, F172014, F182014, F192014, F202014,
        F212014, F222014, F232014, F242014, F252014, F262014, F272014, F282014, F292014, F302014
    ], 1)
}

def Get_F(F_name, dim=30):
    if F_name not in cec_functions:
        raise ValueError(f"Function {F_name} is not a valid CEC 2014 function (F1–F30).")
    
    # Create the function object
    func_obj = cec_functions[F_name](ndim=dim)

    # Return the fitness function, lower bound, upper bound, and dimension
    return func_obj.evaluate, func_obj.lb, func_obj.ub, dim
