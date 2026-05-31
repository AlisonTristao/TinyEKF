#include "TinyEKF.h"
#include <cmath>

static const bool H_MASK[EKF_MEASURE_DIM][EKF_STATE_DIM] = {
    {true, false, false},
    {false, true, false},
    {false, true, true},
    {true, true, false},
    {true, true, false}
};


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

void TinyEKF::predict(const float u[]) {
    // calculating x
    float x0 = EKF_K_L*EKF_TAU_R*u[1];
    float x1 = 0.0050000000000000001*EKF_TAU_L + 0.0050000000000000001*EKF_TAU_R;
    float x2 = EKF_TAU_L - EKF_TAU_R;
    float x3 = EKF_WHEEL_BASE*x[1];
    float x4 = 1/(EKF_TAU_L*EKF_TAU_R);
    float next_x_0 = x4*(0.0050000000000000001*EKF_K_R*EKF_TAU_L*u[0] + EKF_TAU_L*EKF_TAU_R*x[0] + 0.0050000000000000001*x0 - x1*x[0] - 0.0025000000000000001*x2*x3);
    float next_x_1 = -x4*(-0.01*EKF_K_R*EKF_TAU_L*u[0] - EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*x[1] + 0.01*x0 + x1*x3 + 0.01*x2*x[0])/EKF_WHEEL_BASE;
    float next_x_2 = x[2];
    // update x in memory
    x[0] = next_x_0;
    x[1] = next_x_1;
    x[2] = next_x_2;

    // calculating P
    float x0 = pow(EKF_TAU_L, 2);
    float x1 = pow(EKF_TAU_R, 2);
    float x2 = x0*x1;
    float x3 = EKF_TAU_L - EKF_TAU_R;
    float x4 = 0.0025000000000000001*EKF_WHEEL_BASE;
    float x5 = x3*x4;
    float x6 = -EKF_TAU_L*EKF_TAU_R + 0.0050000000000000001*EKF_TAU_L + 0.0050000000000000001*EKF_TAU_R;
    float x7 = P[0][1]*x6 + P[1][1]*x5;
    float x8 = P[0][0]*x6 + P[1][0]*x5;
    float x9 = 1/(x0*x1);
    float x10 = EKF_WHEEL_BASE*x2;
    float x11 = 0.01*x3;
    float x12 = EKF_WHEEL_BASE*x6;
    float x13 = 1.0/EKF_WHEEL_BASE;
    float x14 = x13*x9;
    float x15 = 1.0/EKF_TAU_L;
    float x16 = 0.0050000000000000001*P[0][2];
    float x17 = 1.0/EKF_TAU_R;
    float x18 = P[1][2]*x4;
    float x19 = P[0][1]*x11 + P[1][1]*x12;
    float x20 = P[0][0]*x11 + P[1][0]*x12;
    float x21 = pow(EKF_WHEEL_BASE, 2);
    float x22 = 0.0050000000000000001*P[1][2];
    float x23 = 0.01*x13;
    float x24 = P[0][2]*x23;
    float x25 = 0.0050000000000000001*P[2][0];
    float x26 = P[2][1]*x4;
    float x27 = 0.0050000000000000001*P[2][1];
    float x28 = P[2][0]*x23;
    float next_P_0_0 = x9*(Q[0][0]*x2 + x5*x7 + x6*x8);
    float next_P_0_1 = x14*(Q[0][1]*x10 + x11*x8 + x12*x7);
    float next_P_0_2 = P[0][2] + Q[0][2] - x15*x16 + x15*x18 - x16*x17 - x17*x18;
    float next_P_1_0 = x14*(Q[1][0]*x10 + x19*x5 + x20*x6);
    float next_P_1_1 = x9*(Q[1][1]*x2*x21 + x11*x20 + x12*x19)/x21;
    float next_P_1_2 = P[1][2] + Q[1][2] - x15*x22 + x15*x24 - x17*x22 - x17*x24;
    float next_P_2_0 = P[2][0] + Q[2][0] - x15*x25 + x15*x26 - x17*x25 - x17*x26;
    float next_P_2_1 = P[2][1] + Q[2][1] - x15*x27 + x15*x28 - x17*x27 - x17*x28;
    float next_P_2_2 = P[2][2] + Q[2][2];
    // update P in memory
    P[0][0] = next_P_0_0;
    P[0][1] = next_P_0_1;
    P[0][2] = next_P_0_2;
    P[1][0] = next_P_1_0;
    P[1][1] = next_P_1_1;
    P[1][2] = next_P_1_2;
    P[2][0] = next_P_2_0;
    P[2][1] = next_P_2_1;
    P[2][2] = next_P_2_2;

}

void TinyEKF::update(const float z[], const float u[]) {
    float h[EKF_MEASURE_DIM];
    float H[EKF_MEASURE_DIM][EKF_STATE_DIM];

    // calculating h
    float next_h_0 = x[0];
    float next_h_1 = x[1];
    float next_h_2 = x[1] + x[2];
    float next_h_3 = (1.0/4.0)*(2.0*EKF_K_L*EKF_TAU_R*u[1] + 2.0*EKF_K_R*EKF_TAU_L*u[0] - EKF_WHEEL_BASE*x[1]*(EKF_TAU_L - EKF_TAU_R) - 2*x[0]*(EKF_TAU_L + EKF_TAU_R))/(EKF_TAU_L*EKF_TAU_R);
    float next_h_4 = x[0]*x[1];
    // update h in memory
    h[0] = next_h_0;
    h[1] = next_h_1;
    h[2] = next_h_2;
    h[3] = next_h_3;
    h[4] = next_h_4;

    // calculating H
    float x0 = 1/(EKF_TAU_L*EKF_TAU_R);
    float next_H_0_0 = 1;
    float next_H_0_1 = 0;
    float next_H_0_2 = 0;
    float next_H_1_0 = 0;
    float next_H_1_1 = 1;
    float next_H_1_2 = 0;
    float next_H_2_0 = 0;
    float next_H_2_1 = 1;
    float next_H_2_2 = 1;
    float next_H_3_0 = -1.0/2.0*x0*(EKF_TAU_L + EKF_TAU_R);
    float next_H_3_1 = (1.0/4.0)*EKF_WHEEL_BASE*x0*(-EKF_TAU_L + EKF_TAU_R);
    float next_H_3_2 = 0;
    float next_H_4_0 = x[1];
    float next_H_4_1 = x[0];
    float next_H_4_2 = 0;
    // update H in memory
    H[0][0] = next_H_0_0;
    H[0][1] = next_H_0_1;
    H[0][2] = next_H_0_2;
    H[1][0] = next_H_1_0;
    H[1][1] = next_H_1_1;
    H[1][2] = next_H_1_2;
    H[2][0] = next_H_2_0;
    H[2][1] = next_H_2_1;
    H[2][2] = next_H_2_2;
    H[3][0] = next_H_3_0;
    H[3][1] = next_H_3_1;
    H[3][2] = next_H_3_2;
    H[4][0] = next_H_4_0;
    H[4][1] = next_H_4_1;
    H[4][2] = next_H_4_2;


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
