import sympy as sp
import re

# ==========================================
# 1. parameters and dimensions
# ==========================================
frequency = 100
dt = 1.0 / frequency

dim_x = 2
dim_u = 1
dim_z = 2

# ==========================================
# 2. symbolic variables creation
# ==========================================
def create_sym_matrix(name, rows, cols, is_vector=False):
    if is_vector:
        # corrigido para sp.Matrix e sp.Symbol
        return sp.Matrix(rows, 1, lambda i, j: sp.Symbol(f'{name}[{i}]'))
    else:
        return sp.Matrix(rows, cols, lambda i, j: sp.Symbol(f'{name}[{i}][{j}]'))

# corrigido para True (com maiúscula)
x = create_sym_matrix('x', dim_x, 1, is_vector=True)
u = create_sym_matrix('u', dim_u, 1, is_vector=True) if dim_u > 0 else None
z = create_sym_matrix('z', dim_z, 1, is_vector=True)

p = create_sym_matrix('P', dim_x, dim_x) 
q = create_sym_matrix('Q', dim_x, dim_x) 
r = create_sym_matrix('R', dim_z, dim_z) 
i = sp.eye(dim_x)

# ==========================================
# 3. non-linear physical model (ekf)
# ==========================================
f_eq1 = x[0] + x[1] * dt + 0.5 * sp.cos(x[0]) * u[0]
f_eq2 = x[1] + sp.sin(x[0]) * dt
f = sp.Matrix([f_eq1, f_eq2])

h_eq1 = sp.sqrt(x[0]**2 + x[1]**2)
h_eq2 = sp.sqrt(x[0]**2 + x[1]**2)
h = sp.Matrix([h_eq1, h_eq2])

# ==========================================
# 4. automatic jacobian calculation
# ==========================================
f_jac = f.jacobian(x)
h_jac = h.jacobian(x)

# ==========================================
# 5. header update (.h)
# ==========================================
header_filename = "TinyEKF.h"

try:
    with open(header_filename, "r") as file:
        h_content = file.read()
    h_content = re.sub(r"#define\s+EKF_STATE_DIM\s+\d+", f"#define EKF_STATE_DIM {dim_x}", h_content)
    h_content = re.sub(r"#define\s+EKF_MEASURE_DIM\s+\d+", f"#define EKF_MEASURE_DIM {dim_z}", h_content)
    h_content = re.sub(r"#define\s+EKF_CONTROL_DIM\s+\d+", f"#define EKF_CONTROL_DIM {dim_u}", h_content)
    with open(header_filename, "w") as file:
        file.write(h_content)
    print("file updated successfully!")
except FileNotFoundError:
    print("file not found.")

# ==========================================
# 6. cpp code generation (.cpp)
# ==========================================
cpp_filename = "TinyEKF.cpp"

def generate_cpp_block(expr_matrix, prefix):
    replacements, reduced_exprs = sp.cse(expr_matrix)
    code = f"    // calculating {prefix}\n"
    
    for var, expr in replacements:
        code += f"    float {var} = {sp.ccode(expr)};\n"
    
    reduced_matrix = sp.Matrix(reduced_exprs)
    for i in range(reduced_matrix.rows):
        for j in range(reduced_matrix.cols):
            val = sp.ccode(reduced_matrix[i, j])
            if reduced_matrix.cols == 1:
                code += f"    float next_{prefix}_{i} = {val};\n"
            else:
                code += f"    float next_{prefix}_{i}_{j} = {val};\n"
                
    code += f"    // update {prefix} in memory\n"
    for i in range(expr_matrix.rows):
        for j in range(expr_matrix.cols):
            if expr_matrix.cols == 1:
                code += f"    {prefix}[{i}] = next_{prefix}_{i};\n"
            else:
                code += f"    {prefix}[{i}][{j}] = next_{prefix}_{i}_{j};\n"
    return code

cpp_code = "#include \"TinyEKF.h\"\n"
cpp_code += "#include <cmath>\n\n"

cpp_code += "void TinyEKF::predict(const float u[]) {\n"
cpp_code += generate_cpp_block(f, "x")
cpp_code += generate_cpp_block((f_jac * p * f_jac.T) + q, "Q")
cpp_code += "}\n\n"

cpp_code += "void TinyEKF::update(const float z[]) {\n"
y = z - h
s = (h_jac * p * h_jac.T) + r
k = p * h_jac.T * s.inv()

cpp_code += generate_cpp_block(x + (k * y), "x")
cpp_code += generate_cpp_block((i - (k * h_jac)) * p, "P")
cpp_code += "}\n"

with open(cpp_filename, "w") as file:
    file.write(cpp_code)

print("cpp file generated successfully!")