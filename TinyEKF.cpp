#pragma GCC optimize ("O3,unroll-loops")
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

void TinyEKF::predict(const float u[]) {
    // calculating x (non-linear algebraic block)
    float cse_x_0 = EKF_K_L*EKF_TAU_R*u[1];
    float cse_x_1 = 0.00050000000000000001*EKF_TAU_L + 0.00050000000000000001*EKF_TAU_R;
    float cse_x_2 = EKF_TAU_L - EKF_TAU_R;
    float cse_x_3 = EKF_WHEEL_BASE*x[1];
    float cse_x_4 = 1/(EKF_TAU_L*EKF_TAU_R);
    float next_x_0 = cse_x_4*(0.00050000000000000001*EKF_K_R*EKF_TAU_L*u[0] + EKF_TAU_L*EKF_TAU_R*x[0] + 0.00050000000000000001*cse_x_0 - cse_x_1*x[0] - 0.00025000000000000001*cse_x_2*cse_x_3);
    float next_x_1 = -cse_x_4*(-0.001*EKF_K_R*EKF_TAU_L*u[0] - EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*x[1] + 0.001*cse_x_0 + cse_x_1*cse_x_3 + 0.001*cse_x_2*x[0])/EKF_WHEEL_BASE;
    float next_x_2 = x[2];
    // update x in memory
    x[0] = next_x_0;
    x[1] = next_x_1;
    x[2] = next_x_2;

    float F[EKF_STATE_DIM][EKF_STATE_DIM] = {0};
    // calculating F (non-linear algebraic block)
    float cse_F_0 = 1.0/EKF_TAU_L;
    float cse_F_1 = 1.0/EKF_TAU_R;
    float cse_F_2 = -0.00050000000000000001*cse_F_0 - 0.00050000000000000001*cse_F_1 + 1;
    float cse_F_3 = 1.0/EKF_WHEEL_BASE;
    float next_F_0_0 = cse_F_2;
    float next_F_0_1 = 0.00025000000000000001*EKF_WHEEL_BASE*cse_F_0*cse_F_1*(-EKF_TAU_L + EKF_TAU_R);
    float next_F_0_2 = 0;
    float next_F_1_0 = 0.001*cse_F_3*(cse_F_0 - cse_F_1);
    float next_F_1_1 = cse_F_2;
    float next_F_1_2 = 0;
    float next_F_2_0 = 0;
    float next_F_2_1 = 0;
    float next_F_2_2 = 1;
    // update F in memory
    F[0][0] = next_F_0_0;
    F[0][1] = next_F_0_1;
    F[0][2] = next_F_0_2;
    F[1][0] = next_F_1_0;
    F[1][1] = next_F_1_1;
    F[1][2] = next_F_1_2;
    F[2][0] = next_F_2_0;
    F[2][1] = next_F_2_1;
    F[2][2] = next_F_2_2;

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

void TinyEKF::update(const float z[], const float u[]) {
    float h_vec[EKF_MEASURE_DIM] = {0};
    float H[EKF_MEASURE_DIM][EKF_STATE_DIM] = {0};

    // calculating h_vec (non-linear algebraic block)
    float cse_h_vec_0 = EKF_K_L*EKF_TAU_R*u[1];
    float cse_h_vec_1 = 0.5*EKF_WHEEL_BASE;
    float cse_h_vec_2 = EKF_K_R*EKF_TAU_L*u[0];
    float cse_h_vec_3 = EKF_TAU_L + EKF_TAU_R;
    float cse_h_vec_4 = 2*x[0];
    float cse_h_vec_5 = EKF_TAU_L - EKF_TAU_R;
    float cse_h_vec_6 = EKF_WHEEL_BASE*x[1];
    float cse_h_vec_7 = cse_h_vec_0 - cse_h_vec_2 + (1.0/2.0)*cse_h_vec_3*cse_h_vec_6 + (1.0/2.0)*cse_h_vec_4*cse_h_vec_5;
    float cse_h_vec_8 = 1/(EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE);
    float next_h_vec_0 = x[0];
    float next_h_vec_1 = x[1];
    float next_h_vec_2 = x[1] + x[2];
    float next_h_vec_3 = (1.0/4.0)*cse_h_vec_8*(-4*EKF_IMU_RX*EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*pow(x[1], 2) + 4*EKF_IMU_RY*cse_h_vec_7 - EKF_WHEEL_BASE*(cse_h_vec_3*cse_h_vec_4 + cse_h_vec_5*cse_h_vec_6) + 4*cse_h_vec_0*cse_h_vec_1 + 4*cse_h_vec_1*cse_h_vec_2);
    float next_h_vec_4 = -cse_h_vec_8*(EKF_IMU_RX*cse_h_vec_7 + EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*x[1]*(EKF_IMU_RY*x[1] - x[0]));
    // update h_vec in memory
    h_vec[0] = next_h_vec_0;
    h_vec[1] = next_h_vec_1;
    h_vec[2] = next_h_vec_2;
    h_vec[3] = next_h_vec_3;
    h_vec[4] = next_h_vec_4;

    // calculating H (non-linear algebraic block)
    float cse_H_0 = 1.0/EKF_WHEEL_BASE;
    float cse_H_1 = EKF_TAU_L - EKF_TAU_R;
    float cse_H_2 = 1.0/EKF_TAU_L;
    float cse_H_3 = 1.0/EKF_TAU_R;
    float cse_H_4 = cse_H_2*cse_H_3;
    float cse_H_5 = EKF_IMU_RX*cse_H_0;
    float cse_H_6 = (1.0/2.0)*EKF_IMU_RX;
    float next_H_0_0 = 1;
    float next_H_0_1 = 0;
    float next_H_0_2 = 0;
    float next_H_1_0 = 0;
    float next_H_1_1 = 1;
    float next_H_1_2 = 0;
    float next_H_2_0 = 0;
    float next_H_2_1 = 1;
    float next_H_2_2 = 1;
    float next_H_3_0 = (1.0/2.0)*cse_H_0*cse_H_4*(2*EKF_IMU_RY*cse_H_1 - EKF_TAU_L*EKF_WHEEL_BASE - EKF_TAU_R*EKF_WHEEL_BASE);
    float next_H_3_1 = -1.0/4.0*cse_H_4*(8*EKF_IMU_RX*EKF_TAU_L*EKF_TAU_R*x[1] - 2*EKF_IMU_RY*(EKF_TAU_L + EKF_TAU_R) + EKF_WHEEL_BASE*cse_H_1);
    float next_H_3_2 = 0;
    float next_H_4_0 = cse_H_2*cse_H_5 - cse_H_3*cse_H_5 + x[1];
    float next_H_4_1 = -2*EKF_IMU_RY*x[1] - cse_H_2*cse_H_6 - cse_H_3*cse_H_6 + x[0];
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
