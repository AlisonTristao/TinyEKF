import sympy as sp
import re

# ==========================================
# 1. parameters and dimensions
# ==========================================
frequency = 1000
dt = 1.0 / frequency

dim_x = 3
dim_u = 2
dim_z = 5

USE_IMU_OFFSET = True

# ==========================================
# 2. symbolic constants mapped to header defines
# ==========================================
L = sp.Symbol("EKF_WHEEL_BASE")

K_R = sp.Symbol("EKF_K_R")
K_L = sp.Symbol("EKF_K_L")

Tau_R = sp.Symbol("EKF_TAU_R")
Tau_L = sp.Symbol("EKF_TAU_L")

r_x = sp.Symbol("EKF_IMU_RX")
r_y = sp.Symbol("EKF_IMU_RY")

default_defines = {
    "EKF_STATE_DIM": str(dim_x),
    "EKF_MEASURE_DIM": str(dim_z),
    "EKF_CONTROL_DIM": str(dim_u),

    "EKF_WHEEL_BASE": "0.30f",

    "EKF_K_R": "1.0f",
    "EKF_K_L": "1.0f",

    "EKF_TAU_R": "0.15f",
    "EKF_TAU_L": "0.15f",

    "EKF_IMU_RX": "0.1f",
    "EKF_IMU_RY": "0.1f",
}

# ==========================================
# 3. symbolic variables creation
# ==========================================
def create_sym_matrix(name, rows, cols, is_vector=False):
    if is_vector:
        return sp.Matrix(rows, 1, lambda i, j: sp.Symbol(f'{name}[{i}]'))
    else:
        return sp.Matrix(rows, cols, lambda i, j: sp.Symbol(f'{name}[{i}][{j}]'))

x = create_sym_matrix('x', dim_x, 1, is_vector=True)
u = create_sym_matrix('u', dim_u, 1, is_vector=True)
z = create_sym_matrix('z', dim_z, 1, is_vector=True)

p = create_sym_matrix('P', dim_x, dim_x)
q = create_sym_matrix('Q', dim_x, dim_x)
r = create_sym_matrix('R', dim_z, dim_z)

# ==========================================
# 4. physical model
# x = [v, w, bg]
# u = [u_R, u_L]
# ==========================================
v = x[0]
w = x[1]
bg = x[2]

u_R = u[0]
u_L = u[1]

a_R = 1 / Tau_R
a_L = 1 / Tau_L

b_R = K_R / Tau_R
b_L = K_L / Tau_L

v_dot = (
    -((a_R + a_L) / 2) * v
    -((L * (a_R - a_L)) / 4) * w
    + 0.5 * (b_R * u_R + b_L * u_L)
)

w_dot = (
    -((a_R - a_L) / L) * v
    -((a_R + a_L) / 2) * w
    + (1 / L) * (b_R * u_R - b_L * u_L)
)

f = sp.Matrix([
    v + v_dot * dt,
    w + w_dot * dt,
    bg
])

# ==========================================
# 5. observation model
# z = [v_enc, w_enc, gyro_z, acc_x, acc_y]
# ==========================================
gyro_z = w + bg

if USE_IMU_OFFSET:
    acc_x = v_dot - (w**2) * r_x - w_dot * r_y
    acc_y = v * w + w_dot * r_x - (w**2) * r_y
else:
    acc_x = v_dot
    acc_y = v * w

h = sp.Matrix([
    v,
    w,
    gyro_z,
    acc_x,
    acc_y
])

# ==========================================
# 6. jacobians
# ==========================================
f_jac = sp.simplify(f.jacobian(x))
h_jac = sp.simplify(h.jacobian(x))

# ==========================================
# 7. header update
# ==========================================
header_filename = "TinyEKF.h"

generated_defines = (
    "// ==========================================\n"
    "// ekf generated defines\n"
    "// ==========================================\n"
)

for name, value in default_defines.items():
    generated_defines += f"#define {name} {value}\n"

generated_defines += (
    "// ==========================================\n"
    "// end ekf generated defines\n"
    "// ==========================================\n"
)

try:
    with open(header_filename, "r") as file:
        h_content = file.read()

    pattern = (
        r"// ==========================================\n"
        r"// ekf generated defines\n"
        r"// ==========================================\n"
        r".*?"
        r"// ==========================================\n"
        r"// end ekf generated defines\n"
        r"// ==========================================\n?"
    )

    if re.search(pattern, h_content, flags=re.DOTALL | re.IGNORECASE):
        h_content = re.sub(
            pattern,
            generated_defines,
            h_content,
            flags=re.DOTALL | re.IGNORECASE
        )
    else:
        h_content = generated_defines + "\n" + h_content

    with open(header_filename, "w") as file:
        file.write(h_content)

    print("header file updated successfully!")

except FileNotFoundError:
    with open(header_filename, "w") as file:
        file.write(generated_defines + "\n")
    print("header file created successfully!")

# ==========================================
# 8. cpp generation helpers
# ==========================================
def generate_cpp_block(expr_matrix, prefix):
    expr_matrix = sp.simplify(expr_matrix)
    
    symbols_generator = sp.numbered_symbols(f"cse_{prefix}_")
    replacements, reduced_exprs = sp.cse(expr_matrix, symbols=symbols_generator)

    code = f"    // calculating {prefix} (non-linear algebraic block)\n"

    for var, expr in replacements:
        code += f"    float {var} = {sp.ccode(expr)};\n"

    reduced_matrix = sp.Matrix(reduced_exprs)

    for row in range(reduced_matrix.rows):
        for col in range(reduced_matrix.cols):
            val = sp.ccode(sp.simplify(reduced_matrix[row, col]))

            if reduced_matrix.cols == 1:
                code += f"    float next_{prefix}_{row} = {val};\n"
            else:
                code += f"    float next_{prefix}_{row}_{col} = {val};\n"

    code += f"    // update {prefix} in memory\n"

    for row in range(expr_matrix.rows):
        for col in range(expr_matrix.cols):
            if expr_matrix.cols == 1:
                code += f"    {prefix}[{row}] = next_{prefix}_{row};\n"
            else:
                code += f"    {prefix}[{row}][{col}] = next_{prefix}_{row}_{col};\n"

    return code

def generate_H_mask(H_sym):
    code = "static const bool H_MASK[EKF_MEASURE_DIM][EKF_STATE_DIM] = {\n"

    for row in range(dim_z):
        values = []
        for col in range(dim_x):
            if sp.simplify(H_sym[row, col]) != 0:
                values.append("true")
            else:
                values.append("false")
        
        code += "    {" + ", ".join(values) + "}"
        if row < dim_z - 1:
            code += ","
        code += "\n"

    code += "};\n\n"
    return code

# ==========================================
# 9. cpp code generation
# ==========================================
cpp_filename = "TinyEKF.cpp"

cpp_code = '#pragma GCC optimize ("O3,unroll-loops")\n'
cpp_code += '#include "TinyEKF.h"\n'
cpp_code += "#include <cmath>\n\n"

cpp_code += generate_H_mask(h_jac)

cpp_code += """static bool invertMatrix(
    const float A[EKF_MEASURE_DIM][EKF_MEASURE_DIM],
    float A_inv[EKF_MEASURE_DIM][EKF_MEASURE_DIM]
) {
    float aug[EKF_MEASURE_DIM][2 * EKF_MEASURE_DIM];

    // build augmented matrix
    for (int i = 0; i < EKF_MEASURE_DIM; i++) {
        for (int j = 0; j < EKF_MEASURE_DIM; j++) {
            aug[i][j] = A[i][j];
            aug[i][j + EKF_MEASURE_DIM] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // gauss-jordan elimination
    for (int i = 0; i < EKF_MEASURE_DIM; i++) {
        int pivot = i;
        float max_val = std::fabs(aug[i][i]);

        for (int row = i + 1; row < EKF_MEASURE_DIM; row++) {
            float val = std::fabs(aug[row][i]);
            if (val > max_val) {
                max_val = val;
                pivot = row;
            }
        }

        if (max_val < 1e-9f) {
            return false;
        }

        if (pivot != i) {
            for (int col = 0; col < 2 * EKF_MEASURE_DIM; col++) {
                float temp = aug[i][col];
                aug[i][col] = aug[pivot][col];
                aug[pivot][col] = temp;
            }
        }

        float div = aug[i][i];
        for (int col = 0; col < 2 * EKF_MEASURE_DIM; col++) {
            aug[i][col] /= div;
        }

        for (int row = 0; row < EKF_MEASURE_DIM; row++) {
            if (row == i) continue;
            float factor = aug[row][i];
            for (int col = 0; col < 2 * EKF_MEASURE_DIM; col++) {
                aug[row][col] -= factor * aug[i][col];
            }
        }
    }

    // extract inverse matrix
    for (int i = 0; i < EKF_MEASURE_DIM; i++) {
        for (int j = 0; j < EKF_MEASURE_DIM; j++) {
            A_inv[i][j] = aug[i][j + EKF_MEASURE_DIM];
        }
    }

    return true;
}

"""

# ==========================================
# predict (Hybrid: Symbolic F + loops for P)
# ==========================================
cpp_code += "void TinyEKF::predict(const float u[]) {\n"
cpp_code += generate_cpp_block(f, "x")
cpp_code += "\n    float F[EKF_STATE_DIM][EKF_STATE_DIM] = {0};\n"
cpp_code += generate_cpp_block(f_jac, "F")

cpp_code += """
    float P_next[EKF_STATE_DIM][EKF_STATE_DIM] = {0};
    float Temp[EKF_STATE_DIM][EKF_STATE_DIM] = {0};

    // Temp = F * P
    for (int i = 0; i < EKF_STATE_DIM; i++) {
        for (int j = 0; j < EKF_STATE_DIM; j++) {
            for (int k = 0; k < EKF_STATE_DIM; k++) {
                Temp[i][j] += F[i][k] * P[k][j];
            }
        }
    }

    // P_next = Temp * F^T + Q
    for (int i = 0; i < EKF_STATE_DIM; i++) {
        for (int j = 0; j < EKF_STATE_DIM; j++) {
            for (int k = 0; k < EKF_STATE_DIM; k++) {
                P_next[i][j] += Temp[i][k] * F[j][k]; 
            }
            P_next[i][j] += Q[i][j];
        }
    }

    // Copy back to P
    for (int i = 0; i < EKF_STATE_DIM; i++) {
        for (int j = 0; j < EKF_STATE_DIM; j++) {
            P[i][j] = P_next[i][j];
        }
    }
}

"""

# ==========================================
# update (Hybrid: Symbolic H + loops for Matrices)
# ==========================================
cpp_code += "void TinyEKF::update(const float z[], const float u[]) {\n"
cpp_code += "    float h_vec[EKF_MEASURE_DIM] = {0};\n"
cpp_code += "    float H[EKF_MEASURE_DIM][EKF_STATE_DIM] = {0};\n\n"

cpp_code += generate_cpp_block(h, "h_vec")
cpp_code += "\n"
cpp_code += generate_cpp_block(h_jac, "H")

cpp_code += """
    // 1. Calculate innovation y = z - h_vec
    float y[EKF_MEASURE_DIM];
    for (int i = 0; i < EKF_MEASURE_DIM; i++) {
        y[i] = z[i] - h_vec[i];
    }

    // 2. S = H * P * H^T + R
    float S[EKF_MEASURE_DIM][EKF_MEASURE_DIM] = {0};
    float Temp_HP[EKF_MEASURE_DIM][EKF_STATE_DIM] = {0};

    // Temp_HP = H * P
    for (int i = 0; i < EKF_MEASURE_DIM; i++) {
        for (int j = 0; j < EKF_STATE_DIM; j++) {
            for (int k = 0; k < EKF_STATE_DIM; k++) {
                Temp_HP[i][j] += H[i][k] * P[k][j];
            }
        }
    }

    // S = Temp_HP * H^T + R
    for (int i = 0; i < EKF_MEASURE_DIM; i++) {
        for (int j = 0; j < EKF_MEASURE_DIM; j++) {
            for (int k = 0; k < EKF_STATE_DIM; k++) {
                S[i][j] += Temp_HP[i][k] * H[j][k];
            }
            S[i][j] += R[i][j];
        }
    }

    // 3. Invert S matrix numerically
    float S_inv[EKF_MEASURE_DIM][EKF_MEASURE_DIM];
    if (!invertMatrix(S, S_inv)) {
        return; // Inversion failed, discard measurement
    }

    // 4. K = P * H^T * S_inv
    float K[EKF_STATE_DIM][EKF_MEASURE_DIM] = {0};
    float Temp_PHT[EKF_STATE_DIM][EKF_MEASURE_DIM] = {0};

    // Temp_PHT = P * H^T
    for (int i = 0; i < EKF_STATE_DIM; i++) {
        for (int j = 0; j < EKF_MEASURE_DIM; j++) {
            for (int k = 0; k < EKF_STATE_DIM; k++) {
                Temp_PHT[i][j] += P[i][k] * H[j][k];
            }
        }
    }

    // K = Temp_PHT * S_inv
    for (int i = 0; i < EKF_STATE_DIM; i++) {
        for (int j = 0; j < EKF_MEASURE_DIM; j++) {
            for (int k = 0; k < EKF_MEASURE_DIM; k++) {
                K[i][j] += Temp_PHT[i][k] * S_inv[k][j];
            }
        }
    }

    // 5. Update state estimate: x = x + K * y
    for (int i = 0; i < EKF_STATE_DIM; i++) {
        for (int j = 0; j < EKF_MEASURE_DIM; j++) {
            x[i] += K[i][j] * y[j];
        }
    }

    // 6. Update covariance estimate: P = (I - K * H) * P
    float P_next[EKF_STATE_DIM][EKF_STATE_DIM] = {0};
    
    // P_next = P - K * (H * P) -> Reusing Temp_HP!
    for (int i = 0; i < EKF_STATE_DIM; i++) {
        for (int j = 0; j < EKF_STATE_DIM; j++) {
            P_next[i][j] = P[i][j];
            for (int k = 0; k < EKF_MEASURE_DIM; k++) {
                P_next[i][j] -= K[i][k] * Temp_HP[k][j];
            }
        }
    }

    // Copy back to P
    for (int i = 0; i < EKF_STATE_DIM; i++) {
        for (int j = 0; j < EKF_STATE_DIM; j++) {
            P[i][j] = P_next[i][j];
        }
    }
}
"""

with open(cpp_filename, "w") as file:
    file.write(cpp_code)

print("cpp file generated successfully!")