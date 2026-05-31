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

        // partial pivoting
        for (int row = i + 1; row < EKF_MEASURE_DIM; row++) {
            float val = std::fabs(aug[row][i]);
            if (val > max_val) {
                max_val = val;
                pivot = row;
            }
        }

        // singular matrix check
        if (max_val < 1e-9f) {
            return false;
        }

        // swap rows if needed
        if (pivot != i) {
            for (int col = 0; col < 2 * EKF_MEASURE_DIM; col++) {
                float temp = aug[i][col];
                aug[i][col] = aug[pivot][col];
                aug[pivot][col] = temp;
            }
        }

        // scale pivot row
        float div = aug[i][i];
        for (int col = 0; col < 2 * EKF_MEASURE_DIM; col++) {
            aug[i][col] /= div;
        }

        // eliminate column entries
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

void TinyEKF::predict(const float u[]) {
    // calculating x
    float cse_x_0 = EKF_K_L*EKF_TAU_R*u[1];
    float cse_x_1 = 0.0050000000000000001*EKF_TAU_L + 0.0050000000000000001*EKF_TAU_R;
    float cse_x_2 = EKF_TAU_L - EKF_TAU_R;
    float cse_x_3 = EKF_WHEEL_BASE*x[1];
    float cse_x_4 = 1/(EKF_TAU_L*EKF_TAU_R);
    float next_x_0 = cse_x_4*(0.0050000000000000001*EKF_K_R*EKF_TAU_L*u[0] + EKF_TAU_L*EKF_TAU_R*x[0] + 0.0050000000000000001*cse_x_0 - cse_x_1*x[0] - 0.0025000000000000001*cse_x_2*cse_x_3);
    float next_x_1 = -cse_x_4*(-0.01*EKF_K_R*EKF_TAU_L*u[0] - EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*x[1] + 0.01*cse_x_0 + cse_x_1*cse_x_3 + 0.01*cse_x_2*x[0])/EKF_WHEEL_BASE;
    float next_x_2 = x[2];
    // update x in memory
    x[0] = next_x_0;
    x[1] = next_x_1;
    x[2] = next_x_2;

    // calculating P
    float cse_P_0 = pow(EKF_TAU_L, 2);
    float cse_P_1 = pow(EKF_TAU_R, 2);
    float cse_P_2 = cse_P_0*cse_P_1;
    float cse_P_3 = EKF_TAU_L - EKF_TAU_R;
    float cse_P_4 = 0.0025000000000000001*EKF_WHEEL_BASE;
    float cse_P_5 = cse_P_3*cse_P_4;
    float cse_P_6 = -EKF_TAU_L*EKF_TAU_R + 0.0050000000000000001*EKF_TAU_L + 0.0050000000000000001*EKF_TAU_R;
    float cse_P_7 = P[0][1]*cse_P_6 + P[1][1]*cse_P_5;
    float cse_P_8 = P[0][0]*cse_P_6 + P[1][0]*cse_P_5;
    float cse_P_9 = 1/(cse_P_0*cse_P_1);
    float cse_P_10 = EKF_WHEEL_BASE*cse_P_2;
    float cse_P_11 = 0.01*cse_P_3;
    float cse_P_12 = EKF_WHEEL_BASE*cse_P_6;
    float cse_P_13 = 1.0/EKF_WHEEL_BASE;
    float cse_P_14 = cse_P_13*cse_P_9;
    float cse_P_15 = 1.0/EKF_TAU_L;
    float cse_P_16 = 0.0050000000000000001*P[0][2];
    float cse_P_17 = 1.0/EKF_TAU_R;
    float cse_P_18 = P[1][2]*cse_P_4;
    float cse_P_19 = P[0][1]*cse_P_11 + P[1][1]*cse_P_12;
    float cse_P_20 = P[0][0]*cse_P_11 + P[1][0]*cse_P_12;
    float cse_P_21 = pow(EKF_WHEEL_BASE, 2);
    float cse_P_22 = 0.0050000000000000001*P[1][2];
    float cse_P_23 = 0.01*cse_P_13;
    float cse_P_24 = P[0][2]*cse_P_23;
    float cse_P_25 = 0.0050000000000000001*P[2][0];
    float cse_P_26 = P[2][1]*cse_P_4;
    float cse_P_27 = 0.0050000000000000001*P[2][1];
    float cse_P_28 = P[2][0]*cse_P_23;
    float next_P_0_0 = cse_P_9*(Q[0][0]*cse_P_2 + cse_P_5*cse_P_7 + cse_P_6*cse_P_8);
    float next_P_0_1 = cse_P_14*(Q[0][1]*cse_P_10 + cse_P_11*cse_P_8 + cse_P_12*cse_P_7);
    float next_P_0_2 = P[0][2] + Q[0][2] - cse_P_15*cse_P_16 + cse_P_15*cse_P_18 - cse_P_16*cse_P_17 - cse_P_17*cse_P_18;
    float next_P_1_0 = cse_P_14*(Q[1][0]*cse_P_10 + cse_P_19*cse_P_5 + cse_P_20*cse_P_6);
    float next_P_1_1 = cse_P_9*(Q[1][1]*cse_P_2*cse_P_21 + cse_P_11*cse_P_20 + cse_P_12*cse_P_19)/cse_P_21;
    float next_P_1_2 = P[1][2] + Q[1][2] - cse_P_15*cse_P_22 + cse_P_15*cse_P_24 - cse_P_17*cse_P_22 - cse_P_17*cse_P_24;
    float next_P_2_0 = P[2][0] + Q[2][0] - cse_P_15*cse_P_25 + cse_P_15*cse_P_26 - cse_P_17*cse_P_25 - cse_P_17*cse_P_26;
    float next_P_2_1 = P[2][1] + Q[2][1] - cse_P_15*cse_P_27 + cse_P_15*cse_P_28 - cse_P_17*cse_P_27 - cse_P_17*cse_P_28;
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
    float cse_H_0 = 1/(EKF_TAU_L*EKF_TAU_R);
    float next_H_0_0 = 1;
    float next_H_0_1 = 0;
    float next_H_0_2 = 0;
    float next_H_1_0 = 0;
    float next_H_1_1 = 1;
    float next_H_1_2 = 0;
    float next_H_2_0 = 0;
    float next_H_2_1 = 1;
    float next_H_2_2 = 1;
    float next_H_3_0 = -1.0/2.0*cse_H_0*(EKF_TAU_L + EKF_TAU_R);
    float next_H_3_1 = (1.0/4.0)*EKF_WHEEL_BASE*cse_H_0*(-EKF_TAU_L + EKF_TAU_R);
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

    // calculate innovation y = z - h
    float y[EKF_MEASURE_DIM];
    for (int a = 0; a < EKF_MEASURE_DIM; a++) {
        y[a] = z[a] - h[a];
    }

    // calculate innovation covariance S = H * P * H^T + R
    float S[EKF_MEASURE_DIM][EKF_MEASURE_DIM] = {0};
    for (int a = 0; a < EKF_MEASURE_DIM; a++) {
        for (int b = 0; b < EKF_MEASURE_DIM; b++) {
            S[a][b] = R[a][b];

            for (int c = 0; c < EKF_STATE_DIM; c++) {
                if (!H_MASK[a][c]) continue;

                for (int d = 0; d < EKF_STATE_DIM; d++) {
                    if (!H_MASK[b][d]) continue;
                    S[a][b] += H[a][c] * P[c][d] * H[b][d];
                }
            }
        }
    }

    // invert S matrix
    float S_inv[EKF_MEASURE_DIM][EKF_MEASURE_DIM];
    if (!invertMatrix(S, S_inv)) {
        return; // inversion failed
    }

    // calculate kalman gain K = P * H^T * S_inv
    float K[EKF_STATE_DIM][EKF_MEASURE_DIM] = {0};
    for (int a = 0; a < EKF_STATE_DIM; a++) {
        for (int b = 0; b < EKF_MEASURE_DIM; b++) {
            for (int c = 0; c < EKF_STATE_DIM; c++) {
                for (int d = 0; d < EKF_MEASURE_DIM; d++) {
                    if (!H_MASK[d][c]) continue;
                    K[a][b] += P[a][c] * H[d][c] * S_inv[d][b];
                }
            }
        }
    }

    // update state estimate x = x + K * y
    for (int a = 0; a < EKF_STATE_DIM; a++) {
        float dx = 0.0f;
        for (int b = 0; b < EKF_MEASURE_DIM; b++) {
            dx += K[a][b] * y[b];
        }
        x[a] += dx;
    }

    // calculate KH = K * H
    float KH[EKF_STATE_DIM][EKF_STATE_DIM] = {0};
    for (int a = 0; a < EKF_STATE_DIM; a++) {
        for (int b = 0; b < EKF_STATE_DIM; b++) {
            for (int c = 0; c < EKF_MEASURE_DIM; c++) {
                if (!H_MASK[c][b]) continue;
                KH[a][b] += K[a][c] * H[c][b];
            }
        }
    }

    // calculate I_KH = I - K * H
    float I_KH[EKF_STATE_DIM][EKF_STATE_DIM] = {0};
    for (int a = 0; a < EKF_STATE_DIM; a++) {
        for (int b = 0; b < EKF_STATE_DIM; b++) {
            I_KH[a][b] = (a == b) ? 1.0f - KH[a][b] : -KH[a][b];
        }
    }

    // update covariance estimate P = (I - K * H) * P
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
