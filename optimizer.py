import sympy as sp
import re

# ==========================================
# 1. parameters and dimensions
# ==========================================
frequency = 100
dt = 1.0 / frequency

dim_x = 3
dim_u = 2
dim_z = 5

USE_IMU_OFFSET = False

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

generated_defines = """
// ==========================================
// EKF generated defines
// ==========================================
""".strip()

for name, value in default_defines.items():
    generated_defines += f"\n#define {name} {value}"

generated_defines += """

// ==========================================
// end EKF generated defines
// ==========================================
""".strip("\n")

try:
    with open(header_filename, "r") as file:
        h_content = file.read()

    pattern = (
        r"// ==========================================\n"
        r"// EKF generated defines\n"
        r"// ==========================================\n"
        r".*?"
        r"// ==========================================\n"
        r"// end EKF generated defines\n"
        r"// =========================================="
    )

    if re.search(pattern, h_content, flags=re.DOTALL):
        h_content = re.sub(
            pattern,
            generated_defines,
            h_content,
            flags=re.DOTALL
        )
    else:
        h_content = generated_defines + "\n\n" + h_content

    with open(header_filename, "w") as file:
        file.write(h_content)

    print("header file updated successfully!")

except FileNotFoundError:
    with open(header_filename, "w") as file:
        file.write(generated_defines + "\n\n")
    print("header file created successfully!")

# ==========================================
# 8. cpp generation helpers
# ==========================================
def generate_cpp_block(expr_matrix, prefix):
    expr_matrix = sp.simplify(expr_matrix)
    replacements, reduced_exprs = sp.cse(expr_matrix)

    code = f"    // calculating {prefix}\n"

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

    code += "\n"
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

cpp_code = '#include "TinyEKF.h"\n'
cpp_code += "#include <cmath>\n\n"

cpp_code += generate_H_mask(h_jac)

cpp_code += """
static bool invertMatrix(
    const float A[EKF_MEASURE_DIM][EKF_MEASURE_DIM],
    float A_inv[EKF_MEASURE_DIM][EKF_MEASURE_DIM]
) {
    float aug[EKF_MEASURE_DIM][2 * EKF_MEASURE_DIM];

    for (int i = 0; i < EKF_MEASURE_DIM; i++) {
        for (int j = 0; j < EKF_MEASURE_DIM; j++) {
            aug[i][j] = A[i][j];
            aug[i][j + EKF_MEASURE_DIM] = (i == j) ? 1.0f : 0.0f;
        }
    }

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
            if (row == i) {
                continue;
            }

            float factor = aug[row][i];

            for (int col = 0; col < 2 * EKF_MEASURE_DIM; col++) {
                aug[row][col] -= factor * aug[i][col];
            }
        }
    }

    for (int i = 0; i < EKF_MEASURE_DIM; i++) {
        for (int j = 0; j < EKF_MEASURE_DIM; j++) {
            A_inv[i][j] = aug[i][j + EKF_MEASURE_DIM];
        }
    }

    return true;
}

"""

# ==========================================
# predict
# ==========================================
cpp_code += "void TinyEKF::predict(const float u[]) {\n"

cpp_code += generate_cpp_block(f, "x")

# P = F * P * F^T + Q
cpp_code += generate_cpp_block((f_jac * p * f_jac.T) + q, "P")

cpp_code += "}\n\n"

# ==========================================
# update
# h depends on u because acc_x = v_dot
# ==========================================
cpp_code += "void TinyEKF::update(const float z[], const float u[]) {\n"

cpp_code += "    float h[EKF_MEASURE_DIM];\n"
cpp_code += "    float H[EKF_MEASURE_DIM][EKF_STATE_DIM];\n\n"

cpp_code += generate_cpp_block(h, "h")
cpp_code += generate_cpp_block(h_jac, "H")

cpp_code += """
    // y = z - h
    float y[EKF_MEASURE_DIM];

    for (int a = 0; a < EKF_MEASURE_DIM; a++) {
        y[a] = z[a] - h[a];
    }

    // S = H * P * H^T + R
    float S[EKF_MEASURE_DIM][EKF_MEASURE_DIM] = {0};

    for (int a = 0; a < EKF_MEASURE_DIM; a++) {
        for (int b = 0; b < EKF_MEASURE_DIM; b++) {
            S[a][b] = R[a][b];

            for (int c = 0; c < EKF_STATE_DIM; c++) {
                if (!H_MASK[a][c]) {
                    continue;
                }

                for (int d = 0; d < EKF_STATE_DIM; d++) {
                    if (!H_MASK[b][d]) {
                        continue;
                    }

                    S[a][b] += H[a][c] * P[c][d] * H[b][d];
                }
            }
        }
    }

    // S_inv = inv(S)
    float S_inv[EKF_MEASURE_DIM][EKF_MEASURE_DIM];

    if (!invertMatrix(S, S_inv)) {
        return;
    }

    // K = P * H^T * S_inv
    float K[EKF_STATE_DIM][EKF_MEASURE_DIM] = {0};

    for (int a = 0; a < EKF_STATE_DIM; a++) {
        for (int b = 0; b < EKF_MEASURE_DIM; b++) {
            for (int c = 0; c < EKF_STATE_DIM; c++) {
                for (int d = 0; d < EKF_MEASURE_DIM; d++) {
                    if (!H_MASK[d][c]) {
                        continue;
                    }

                    K[a][b] += P[a][c] * H[d][c] * S_inv[d][b];
                }
            }
        }
    }

    // x = x + K * y
    for (int a = 0; a < EKF_STATE_DIM; a++) {
        float dx = 0.0f;

        for (int b = 0; b < EKF_MEASURE_DIM; b++) {
            dx += K[a][b] * y[b];
        }

        x[a] += dx;
    }

    // KH = K * H
    float KH[EKF_STATE_DIM][EKF_STATE_DIM] = {0};

    for (int a = 0; a < EKF_STATE_DIM; a++) {
        for (int b = 0; b < EKF_STATE_DIM; b++) {
            for (int c = 0; c < EKF_MEASURE_DIM; c++) {
                if (!H_MASK[c][b]) {
                    continue;
                }

                KH[a][b] += K[a][c] * H[c][b];
            }
        }
    }

    // I_KH = I - K * H
    float I_KH[EKF_STATE_DIM][EKF_STATE_DIM] = {0};

    for (int a = 0; a < EKF_STATE_DIM; a++) {
        for (int b = 0; b < EKF_STATE_DIM; b++) {
            I_KH[a][b] = -KH[a][b];

            if (a == b) {
                I_KH[a][b] += 1.0f;
            }
        }
    }

    // P = (I - K * H) * P
    float newP[EKF_STATE_DIM][EKF_STATE_DIM] = {0};

    for (int a = 0; a < EKF_STATE_DIM; a++) {
        for (int b = 0; b < EKF_STATE_DIM; b++) {
            for (int c = 0; c < EKF_STATE_DIM; c++) {
                newP[a][b] += I_KH[a][c] * P[c][b];
            }
        }
    }

    for (int a = 0; a < EKF_STATE_DIM; a++) {
        for (int b = 0; b < EKF_STATE_DIM; b++) {
            P[a][b] = newP[a][b];
        }
    }
}
"""

with open(cpp_filename, "w") as file:
    file.write(cpp_code)

print("cpp file generated successfully!")