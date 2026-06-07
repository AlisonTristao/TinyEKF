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

    // calculating P
    float cse_P_0 = pow(EKF_TAU_L, 2);
    float cse_P_1 = pow(EKF_TAU_R, 2);
    float cse_P_2 = cse_P_0*cse_P_1;
    float cse_P_3 = EKF_TAU_L - EKF_TAU_R;
    float cse_P_4 = 0.00025000000000000001*EKF_WHEEL_BASE;
    float cse_P_5 = cse_P_3*cse_P_4;
    float cse_P_6 = -EKF_TAU_L*EKF_TAU_R + 0.00050000000000000001*EKF_TAU_L + 0.00050000000000000001*EKF_TAU_R;
    float cse_P_7 = P[0][1]*cse_P_6 + P[1][1]*cse_P_5;
    float cse_P_8 = P[0][0]*cse_P_6 + P[1][0]*cse_P_5;
    float cse_P_9 = 1/(cse_P_0*cse_P_1);
    float cse_P_10 = EKF_WHEEL_BASE*cse_P_2;
    float cse_P_11 = 0.001*cse_P_3;
    float cse_P_12 = EKF_WHEEL_BASE*cse_P_6;
    float cse_P_13 = 1.0/EKF_WHEEL_BASE;
    float cse_P_14 = cse_P_13*cse_P_9;
    float cse_P_15 = 1.0/EKF_TAU_L;
    float cse_P_16 = 0.00050000000000000001*P[0][2];
    float cse_P_17 = 1.0/EKF_TAU_R;
    float cse_P_18 = P[1][2]*cse_P_4;
    float cse_P_19 = P[0][1]*cse_P_11 + P[1][1]*cse_P_12;
    float cse_P_20 = P[0][0]*cse_P_11 + P[1][0]*cse_P_12;
    float cse_P_21 = pow(EKF_WHEEL_BASE, 2);
    float cse_P_22 = 0.00050000000000000001*P[1][2];
    float cse_P_23 = 0.001*cse_P_13;
    float cse_P_24 = P[0][2]*cse_P_23;
    float cse_P_25 = 0.00050000000000000001*P[2][0];
    float cse_P_26 = P[2][1]*cse_P_4;
    float cse_P_27 = 0.00050000000000000001*P[2][1];
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
    float cse_h_0 = EKF_K_L*EKF_TAU_R*u[1];
    float cse_h_1 = 0.5*EKF_WHEEL_BASE;
    float cse_h_2 = EKF_K_R*EKF_TAU_L*u[0];
    float cse_h_3 = EKF_TAU_L + EKF_TAU_R;
    float cse_h_4 = 2*x[0];
    float cse_h_5 = EKF_TAU_L - EKF_TAU_R;
    float cse_h_6 = EKF_WHEEL_BASE*x[1];
    float cse_h_7 = cse_h_0 - cse_h_2 + (1.0/2.0)*cse_h_3*cse_h_6 + (1.0/2.0)*cse_h_4*cse_h_5;
    float cse_h_8 = 1/(EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE);
    float next_h_0 = x[0];
    float next_h_1 = x[1];
    float next_h_2 = x[1] + x[2];
    float next_h_3 = (1.0/4.0)*cse_h_8*(-4*EKF_IMU_RX*EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*pow(x[1], 2) + 4*EKF_IMU_RY*cse_h_7 - EKF_WHEEL_BASE*(cse_h_3*cse_h_4 + cse_h_5*cse_h_6) + 4*cse_h_0*cse_h_1 + 4*cse_h_1*cse_h_2);
    float next_h_4 = -cse_h_8*(EKF_IMU_RX*cse_h_7 + EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*x[1]*(EKF_IMU_RY*x[1] - x[0]));
    // update h in memory
    h[0] = next_h_0;
    h[1] = next_h_1;
    h[2] = next_h_2;
    h[3] = next_h_3;
    h[4] = next_h_4;

    // calculating H
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

    // calculate innovation y = z - h
    float y[EKF_MEASURE_DIM];
    for (int a = 0; a < EKF_MEASURE_DIM; a++) {
        y[a] = z[a] - h[a];
    }

    // declare S matrix
    float S[EKF_MEASURE_DIM][EKF_MEASURE_DIM] = {0};

    // calculating S
    float cse_S_0 = P[0][1] + P[0][2];
    float cse_S_1 = EKF_TAU_L*EKF_WHEEL_BASE;
    float cse_S_2 = EKF_TAU_L - EKF_TAU_R;
    float cse_S_3 = 2*EKF_IMU_RY;
    float cse_S_4 = EKF_TAU_R*EKF_WHEEL_BASE + cse_S_1 - cse_S_2*cse_S_3;
    float cse_S_5 = (1.0/2.0)*cse_S_4;
    float cse_S_6 = P[0][0]*cse_S_5;
    float cse_S_7 = EKF_IMU_RX*EKF_TAU_R;
    float cse_S_8 = EKF_WHEEL_BASE*(8*EKF_TAU_L*cse_S_7*x[1] + EKF_WHEEL_BASE*cse_S_2 - cse_S_3*(EKF_TAU_L + EKF_TAU_R));
    float cse_S_9 = (1.0/4.0)*cse_S_8;
    float cse_S_10 = 1.0/EKF_TAU_R;
    float cse_S_11 = 1.0/EKF_TAU_L;
    float cse_S_12 = 1.0/EKF_WHEEL_BASE;
    float cse_S_13 = cse_S_10*cse_S_11*cse_S_12;
    float cse_S_14 = (1.0/2.0)*EKF_IMU_RX;
    float cse_S_15 = P[0][1]*cse_S_14;
    float cse_S_16 = cse_S_10*cse_S_12;
    float cse_S_17 = -EKF_IMU_RX*P[0][0]*cse_S_11*cse_S_12 + EKF_IMU_RX*P[0][0]*cse_S_16 - P[0][0]*x[1];
    float cse_S_18 = P[1][1] + P[1][2];
    float cse_S_19 = P[1][1]*cse_S_8;
    float cse_S_20 = (1.0/4.0)*cse_S_19;
    float cse_S_21 = P[1][1]*cse_S_14;
    float cse_S_22 = cse_S_3*x[1];
    float cse_S_23 = P[1][1]*cse_S_22 - P[1][1]*x[0] + cse_S_10*cse_S_21 + cse_S_11*cse_S_21;
    float cse_S_24 = P[1][0] + P[2][0];
    float cse_S_25 = P[1][1] + P[2][1];
    float cse_S_26 = EKF_TAU_R*cse_S_1;
    float cse_S_27 = EKF_IMU_RX*EKF_TAU_L;
    float cse_S_28 = cse_S_26*x[1] - cse_S_27 + cse_S_7;
    float cse_S_29 = EKF_WHEEL_BASE*(2*EKF_TAU_L*EKF_TAU_R*(cse_S_22 - x[0]) + cse_S_27 + cse_S_7);
    float cse_S_30 = (1.0/2.0)*cse_S_29;
    float cse_S_31 = P[1][0]*cse_S_8;
    float cse_S_32 = pow(EKF_TAU_L, 2);
    float cse_S_33 = pow(EKF_TAU_R, 2);
    float cse_S_34 = pow(EKF_WHEEL_BASE, 2);
    float cse_S_35 = cse_S_32*cse_S_33*cse_S_34;
    float cse_S_36 = 2*cse_S_4;
    float cse_S_37 = P[0][0]*cse_S_36 + cse_S_31;
    float cse_S_38 = P[0][1]*cse_S_36 + cse_S_19;
    float cse_S_39 = 1/(cse_S_32*cse_S_33*cse_S_34);
    float cse_S_40 = P[1][0]*cse_S_14;
    float cse_S_41 = 2*cse_S_28;
    float cse_S_42 = -P[0][0]*cse_S_41 + P[1][0]*cse_S_29;
    float cse_S_43 = -P[0][1]*cse_S_41 + P[1][1]*cse_S_29;
    float next_S_0_0 = P[0][0] + R[0][0];
    float next_S_0_1 = P[0][1] + R[0][1];
    float next_S_0_2 = R[0][2] + cse_S_0;
    float next_S_0_3 = cse_S_13*(EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*R[0][3] - P[0][1]*cse_S_9 - cse_S_6);
    float next_S_0_4 = -P[0][1]*cse_S_3*x[1] + P[0][1]*x[0] + R[0][4] - cse_S_10*cse_S_15 - cse_S_11*cse_S_15 - cse_S_17;
    float next_S_1_0 = P[1][0] + R[1][0];
    float next_S_1_1 = P[1][1] + R[1][1];
    float next_S_1_2 = R[1][2] + cse_S_18;
    float next_S_1_3 = cse_S_13*(EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*R[1][3] - P[1][0]*cse_S_5 - cse_S_20);
    float next_S_1_4 = EKF_IMU_RX*P[1][0]*cse_S_11*cse_S_12 - EKF_IMU_RX*P[1][0]*cse_S_16 + P[1][0]*x[1] + R[1][4] - cse_S_23;
    float next_S_2_0 = R[2][0] + cse_S_24;
    float next_S_2_1 = R[2][1] + cse_S_25;
    float next_S_2_2 = P[1][2] + P[2][2] + R[2][2] + cse_S_25;
    float next_S_2_3 = cse_S_13*(EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*R[2][3] - cse_S_24*cse_S_5 - cse_S_25*cse_S_9);
    float next_S_2_4 = cse_S_13*(R[2][4]*cse_S_26 + cse_S_24*cse_S_28 - cse_S_25*cse_S_30);
    float next_S_3_0 = (1.0/4.0)*cse_S_13*(4*EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*R[3][0] - cse_S_31 - 4*cse_S_6);
    float next_S_3_1 = cse_S_13*(EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*R[3][1] - P[0][1]*cse_S_5 - cse_S_20);
    float next_S_3_2 = cse_S_13*(EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*R[3][2] - cse_S_0*cse_S_5 - cse_S_18*cse_S_9);
    float next_S_3_3 = (1.0/16.0)*cse_S_39*(16*R[3][3]*cse_S_35 + 2*cse_S_37*cse_S_4 + cse_S_38*cse_S_8);
    float next_S_3_4 = (1.0/8.0)*cse_S_39*(8*R[3][4]*cse_S_35 - 2*cse_S_28*cse_S_37 + cse_S_29*cse_S_38);
    float next_S_4_0 = -P[1][0]*cse_S_3*x[1] + P[1][0]*x[0] + R[4][0] - cse_S_10*cse_S_40 - cse_S_11*cse_S_40 - cse_S_17;
    float next_S_4_1 = EKF_IMU_RX*P[0][1]*cse_S_11*cse_S_12 - EKF_IMU_RX*P[0][1]*cse_S_16 + P[0][1]*x[1] + R[4][1] - cse_S_23;
    float next_S_4_2 = cse_S_13*(R[4][2]*cse_S_26 + cse_S_0*cse_S_28 - cse_S_18*cse_S_30);
    float next_S_4_3 = (1.0/8.0)*cse_S_39*(8*R[4][3]*cse_S_35 + 2*cse_S_4*cse_S_42 + cse_S_43*cse_S_8);
    float next_S_4_4 = (1.0/4.0)*cse_S_39*(4*R[4][4]*cse_S_35 - 2*cse_S_28*cse_S_42 + cse_S_29*cse_S_43);
    // update S in memory
    S[0][0] = next_S_0_0;
    S[0][1] = next_S_0_1;
    S[0][2] = next_S_0_2;
    S[0][3] = next_S_0_3;
    S[0][4] = next_S_0_4;
    S[1][0] = next_S_1_0;
    S[1][1] = next_S_1_1;
    S[1][2] = next_S_1_2;
    S[1][3] = next_S_1_3;
    S[1][4] = next_S_1_4;
    S[2][0] = next_S_2_0;
    S[2][1] = next_S_2_1;
    S[2][2] = next_S_2_2;
    S[2][3] = next_S_2_3;
    S[2][4] = next_S_2_4;
    S[3][0] = next_S_3_0;
    S[3][1] = next_S_3_1;
    S[3][2] = next_S_3_2;
    S[3][3] = next_S_3_3;
    S[3][4] = next_S_3_4;
    S[4][0] = next_S_4_0;
    S[4][1] = next_S_4_1;
    S[4][2] = next_S_4_2;
    S[4][3] = next_S_4_3;
    S[4][4] = next_S_4_4;

    // invert S matrix numerically (prevents instruction cache miss from huge symbolic inverse)
    float S_inv[EKF_MEASURE_DIM][EKF_MEASURE_DIM];
    if (!invertMatrix(S, S_inv)) {
        return; // inversion failed, discard measurement
    }

    // declare K matrix
    float K[EKF_STATE_DIM][EKF_MEASURE_DIM] = {0};

    // calculating K
    float cse_K_0 = P[0][1] + P[0][2];
    float cse_K_1 = EKF_IMU_RX*EKF_TAU_R;
    float cse_K_2 = EKF_IMU_RX*EKF_TAU_L;
    float cse_K_3 = EKF_TAU_L*EKF_WHEEL_BASE;
    float cse_K_4 = EKF_TAU_R*cse_K_3*x[1] + cse_K_1 - cse_K_2;
    float cse_K_5 = 2*P[0][0];
    float cse_K_6 = 2*EKF_IMU_RY;
    float cse_K_7 = 2*EKF_TAU_L*EKF_TAU_R*(cse_K_6*x[1] - x[0]) + cse_K_1 + cse_K_2;
    float cse_K_8 = EKF_WHEEL_BASE*P[0][1];
    float cse_K_9 = -1.0/2.0*cse_K_4*cse_K_5 + (1.0/2.0)*cse_K_7*cse_K_8;
    float cse_K_10 = EKF_TAU_L - EKF_TAU_R;
    float cse_K_11 = EKF_TAU_R*EKF_WHEEL_BASE - cse_K_10*cse_K_6 + cse_K_3;
    float cse_K_12 = 8*EKF_TAU_L*cse_K_1*x[1] + EKF_WHEEL_BASE*cse_K_10 - cse_K_6*(EKF_TAU_L + EKF_TAU_R);
    float cse_K_13 = (1.0/4.0)*cse_K_11*cse_K_5 + (1.0/4.0)*cse_K_12*cse_K_8;
    float cse_K_14 = 1/(EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE);
    float cse_K_15 = P[1][1] + P[1][2];
    float cse_K_16 = 2*P[1][0];
    float cse_K_17 = EKF_WHEEL_BASE*P[1][1];
    float cse_K_18 = -1.0/2.0*cse_K_16*cse_K_4 + (1.0/2.0)*cse_K_17*cse_K_7;
    float cse_K_19 = (1.0/4.0)*cse_K_11*cse_K_16 + (1.0/4.0)*cse_K_12*cse_K_17;
    float cse_K_20 = P[2][1] + P[2][2];
    float cse_K_21 = 2*P[2][0];
    float cse_K_22 = EKF_WHEEL_BASE*P[2][1];
    float cse_K_23 = -1.0/2.0*cse_K_21*cse_K_4 + (1.0/2.0)*cse_K_22*cse_K_7;
    float cse_K_24 = (1.0/4.0)*cse_K_11*cse_K_21 + (1.0/4.0)*cse_K_12*cse_K_22;
    float next_K_0_0 = -cse_K_14*(-EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*(P[0][0]*S_inv[0][0] + P[0][1]*S_inv[1][0] + S_inv[2][0]*cse_K_0) + S_inv[3][0]*cse_K_13 + S_inv[4][0]*cse_K_9);
    float next_K_0_1 = -cse_K_14*(-EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*(P[0][0]*S_inv[0][1] + P[0][1]*S_inv[1][1] + S_inv[2][1]*cse_K_0) + S_inv[3][1]*cse_K_13 + S_inv[4][1]*cse_K_9);
    float next_K_0_2 = -cse_K_14*(-EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*(P[0][0]*S_inv[0][2] + P[0][1]*S_inv[1][2] + S_inv[2][2]*cse_K_0) + S_inv[3][2]*cse_K_13 + S_inv[4][2]*cse_K_9);
    float next_K_0_3 = -cse_K_14*(-EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*(P[0][0]*S_inv[0][3] + P[0][1]*S_inv[1][3] + S_inv[2][3]*cse_K_0) + S_inv[3][3]*cse_K_13 + S_inv[4][3]*cse_K_9);
    float next_K_0_4 = -cse_K_14*(-EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*(P[0][0]*S_inv[0][4] + P[0][1]*S_inv[1][4] + S_inv[2][4]*cse_K_0) + S_inv[3][4]*cse_K_13 + S_inv[4][4]*cse_K_9);
    float next_K_1_0 = -cse_K_14*(-EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*(P[1][0]*S_inv[0][0] + P[1][1]*S_inv[1][0] + S_inv[2][0]*cse_K_15) + S_inv[3][0]*cse_K_19 + S_inv[4][0]*cse_K_18);
    float next_K_1_1 = -cse_K_14*(-EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*(P[1][0]*S_inv[0][1] + P[1][1]*S_inv[1][1] + S_inv[2][1]*cse_K_15) + S_inv[3][1]*cse_K_19 + S_inv[4][1]*cse_K_18);
    float next_K_1_2 = -cse_K_14*(-EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*(P[1][0]*S_inv[0][2] + P[1][1]*S_inv[1][2] + S_inv[2][2]*cse_K_15) + S_inv[3][2]*cse_K_19 + S_inv[4][2]*cse_K_18);
    float next_K_1_3 = -cse_K_14*(-EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*(P[1][0]*S_inv[0][3] + P[1][1]*S_inv[1][3] + S_inv[2][3]*cse_K_15) + S_inv[3][3]*cse_K_19 + S_inv[4][3]*cse_K_18);
    float next_K_1_4 = -cse_K_14*(-EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*(P[1][0]*S_inv[0][4] + P[1][1]*S_inv[1][4] + S_inv[2][4]*cse_K_15) + S_inv[3][4]*cse_K_19 + S_inv[4][4]*cse_K_18);
    float next_K_2_0 = -cse_K_14*(-EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*(P[2][0]*S_inv[0][0] + P[2][1]*S_inv[1][0] + S_inv[2][0]*cse_K_20) + S_inv[3][0]*cse_K_24 + S_inv[4][0]*cse_K_23);
    float next_K_2_1 = -cse_K_14*(-EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*(P[2][0]*S_inv[0][1] + P[2][1]*S_inv[1][1] + S_inv[2][1]*cse_K_20) + S_inv[3][1]*cse_K_24 + S_inv[4][1]*cse_K_23);
    float next_K_2_2 = -cse_K_14*(-EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*(P[2][0]*S_inv[0][2] + P[2][1]*S_inv[1][2] + S_inv[2][2]*cse_K_20) + S_inv[3][2]*cse_K_24 + S_inv[4][2]*cse_K_23);
    float next_K_2_3 = -cse_K_14*(-EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*(P[2][0]*S_inv[0][3] + P[2][1]*S_inv[1][3] + S_inv[2][3]*cse_K_20) + S_inv[3][3]*cse_K_24 + S_inv[4][3]*cse_K_23);
    float next_K_2_4 = -cse_K_14*(-EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*(P[2][0]*S_inv[0][4] + P[2][1]*S_inv[1][4] + S_inv[2][4]*cse_K_20) + S_inv[3][4]*cse_K_24 + S_inv[4][4]*cse_K_23);
    // update K in memory
    K[0][0] = next_K_0_0;
    K[0][1] = next_K_0_1;
    K[0][2] = next_K_0_2;
    K[0][3] = next_K_0_3;
    K[0][4] = next_K_0_4;
    K[1][0] = next_K_1_0;
    K[1][1] = next_K_1_1;
    K[1][2] = next_K_1_2;
    K[1][3] = next_K_1_3;
    K[1][4] = next_K_1_4;
    K[2][0] = next_K_2_0;
    K[2][1] = next_K_2_1;
    K[2][2] = next_K_2_2;
    K[2][3] = next_K_2_3;
    K[2][4] = next_K_2_4;

    // update state estimate x
    // calculating x
    float cse_x_0 = P[0][1] + P[0][2];
    float cse_x_1 = EKF_TAU_L*EKF_WHEEL_BASE;
    float cse_x_2 = EKF_TAU_R*cse_x_1;
    float cse_x_3 = 4*cse_x_2;
    float cse_x_4 = EKF_IMU_RX*EKF_TAU_R;
    float cse_x_5 = EKF_IMU_RX*EKF_TAU_L;
    float cse_x_6 = cse_x_2*x[1] + cse_x_4 - cse_x_5;
    float cse_x_7 = 2*P[0][0];
    float cse_x_8 = 2*EKF_IMU_RY;
    float cse_x_9 = 2*EKF_TAU_L*EKF_TAU_R*(cse_x_8*x[1] - x[0]) + cse_x_4 + cse_x_5;
    float cse_x_10 = EKF_WHEEL_BASE*P[0][1];
    float cse_x_11 = 2*cse_x_10*cse_x_9 - 2*cse_x_6*cse_x_7;
    float cse_x_12 = EKF_TAU_L - EKF_TAU_R;
    float cse_x_13 = EKF_TAU_R*EKF_WHEEL_BASE + cse_x_1 - cse_x_12*cse_x_8;
    float cse_x_14 = 8*EKF_TAU_L*cse_x_4*x[1] + EKF_WHEEL_BASE*cse_x_12 - cse_x_8*(EKF_TAU_L + EKF_TAU_R);
    float cse_x_15 = cse_x_10*cse_x_14 + cse_x_13*cse_x_7;
    float cse_x_16 = (1.0/4.0)/(EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE);
    float cse_x_17 = P[1][1] + P[1][2];
    float cse_x_18 = 2*P[1][0];
    float cse_x_19 = EKF_WHEEL_BASE*P[1][1];
    float cse_x_20 = -2*cse_x_18*cse_x_6 + 2*cse_x_19*cse_x_9;
    float cse_x_21 = cse_x_13*cse_x_18 + cse_x_14*cse_x_19;
    float cse_x_22 = P[2][1] + P[2][2];
    float cse_x_23 = 2*P[2][0];
    float cse_x_24 = EKF_WHEEL_BASE*P[2][1];
    float cse_x_25 = -2*cse_x_23*cse_x_6 + 2*cse_x_24*cse_x_9;
    float cse_x_26 = cse_x_13*cse_x_23 + cse_x_14*cse_x_24;
    float next_x_0 = -cse_x_16*(-4*EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*x[0] + y[0]*(S_inv[3][0]*cse_x_15 + S_inv[4][0]*cse_x_11 - cse_x_3*(P[0][0]*S_inv[0][0] + P[0][1]*S_inv[1][0] + S_inv[2][0]*cse_x_0)) + y[1]*(S_inv[3][1]*cse_x_15 + S_inv[4][1]*cse_x_11 - cse_x_3*(P[0][0]*S_inv[0][1] + P[0][1]*S_inv[1][1] + S_inv[2][1]*cse_x_0)) + y[2]*(S_inv[3][2]*cse_x_15 + S_inv[4][2]*cse_x_11 - cse_x_3*(P[0][0]*S_inv[0][2] + P[0][1]*S_inv[1][2] + S_inv[2][2]*cse_x_0)) + y[3]*(S_inv[3][3]*cse_x_15 + S_inv[4][3]*cse_x_11 - cse_x_3*(P[0][0]*S_inv[0][3] + P[0][1]*S_inv[1][3] + S_inv[2][3]*cse_x_0)) + y[4]*(S_inv[3][4]*cse_x_15 + S_inv[4][4]*cse_x_11 - cse_x_3*(P[0][0]*S_inv[0][4] + P[0][1]*S_inv[1][4] + S_inv[2][4]*cse_x_0)));
    float next_x_1 = -cse_x_16*(-4*EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*x[1] + y[0]*(S_inv[3][0]*cse_x_21 + S_inv[4][0]*cse_x_20 - cse_x_3*(P[1][0]*S_inv[0][0] + P[1][1]*S_inv[1][0] + S_inv[2][0]*cse_x_17)) + y[1]*(S_inv[3][1]*cse_x_21 + S_inv[4][1]*cse_x_20 - cse_x_3*(P[1][0]*S_inv[0][1] + P[1][1]*S_inv[1][1] + S_inv[2][1]*cse_x_17)) + y[2]*(S_inv[3][2]*cse_x_21 + S_inv[4][2]*cse_x_20 - cse_x_3*(P[1][0]*S_inv[0][2] + P[1][1]*S_inv[1][2] + S_inv[2][2]*cse_x_17)) + y[3]*(S_inv[3][3]*cse_x_21 + S_inv[4][3]*cse_x_20 - cse_x_3*(P[1][0]*S_inv[0][3] + P[1][1]*S_inv[1][3] + S_inv[2][3]*cse_x_17)) + y[4]*(S_inv[3][4]*cse_x_21 + S_inv[4][4]*cse_x_20 - cse_x_3*(P[1][0]*S_inv[0][4] + P[1][1]*S_inv[1][4] + S_inv[2][4]*cse_x_17)));
    float next_x_2 = -cse_x_16*(-4*EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*x[2] + y[0]*(S_inv[3][0]*cse_x_26 + S_inv[4][0]*cse_x_25 - cse_x_3*(P[2][0]*S_inv[0][0] + P[2][1]*S_inv[1][0] + S_inv[2][0]*cse_x_22)) + y[1]*(S_inv[3][1]*cse_x_26 + S_inv[4][1]*cse_x_25 - cse_x_3*(P[2][0]*S_inv[0][1] + P[2][1]*S_inv[1][1] + S_inv[2][1]*cse_x_22)) + y[2]*(S_inv[3][2]*cse_x_26 + S_inv[4][2]*cse_x_25 - cse_x_3*(P[2][0]*S_inv[0][2] + P[2][1]*S_inv[1][2] + S_inv[2][2]*cse_x_22)) + y[3]*(S_inv[3][3]*cse_x_26 + S_inv[4][3]*cse_x_25 - cse_x_3*(P[2][0]*S_inv[0][3] + P[2][1]*S_inv[1][3] + S_inv[2][3]*cse_x_22)) + y[4]*(S_inv[3][4]*cse_x_26 + S_inv[4][4]*cse_x_25 - cse_x_3*(P[2][0]*S_inv[0][4] + P[2][1]*S_inv[1][4] + S_inv[2][4]*cse_x_22)));
    // update x in memory
    x[0] = next_x_0;
    x[1] = next_x_1;
    x[2] = next_x_2;

    // update covariance estimate P
    // calculating P
    float cse_P_0 = P[0][1] + P[0][2];
    float cse_P_1 = EKF_TAU_L*EKF_WHEEL_BASE;
    float cse_P_2 = EKF_TAU_R*cse_P_1;
    float cse_P_3 = 4*cse_P_2;
    float cse_P_4 = EKF_IMU_RX*EKF_TAU_R;
    float cse_P_5 = EKF_IMU_RX*EKF_TAU_L;
    float cse_P_6 = cse_P_2*x[1] + cse_P_4 - cse_P_5;
    float cse_P_7 = 2*P[0][0];
    float cse_P_8 = 2*EKF_IMU_RY;
    float cse_P_9 = EKF_TAU_L*EKF_TAU_R;
    float cse_P_10 = cse_P_4 + cse_P_5 + 2*cse_P_9*(cse_P_8*x[1] - x[0]);
    float cse_P_11 = EKF_WHEEL_BASE*P[0][1];
    float cse_P_12 = 2*cse_P_10*cse_P_11 - 2*cse_P_6*cse_P_7;
    float cse_P_13 = EKF_TAU_L - EKF_TAU_R;
    float cse_P_14 = EKF_TAU_R*EKF_WHEEL_BASE + cse_P_1 - cse_P_13*cse_P_8;
    float cse_P_15 = 8*EKF_TAU_L*cse_P_4*x[1] + EKF_WHEEL_BASE*cse_P_13 - cse_P_8*(EKF_TAU_L + EKF_TAU_R);
    float cse_P_16 = cse_P_11*cse_P_15 + cse_P_14*cse_P_7;
    float cse_P_17 = S_inv[3][2]*cse_P_16 + S_inv[4][2]*cse_P_12 - cse_P_3*(P[0][0]*S_inv[0][2] + P[0][1]*S_inv[1][2] + S_inv[2][2]*cse_P_0);
    float cse_P_18 = cse_P_17*cse_P_3;
    float cse_P_19 = pow(EKF_WHEEL_BASE, 2);
    float cse_P_20 = pow(EKF_TAU_L, 2);
    float cse_P_21 = pow(EKF_TAU_R, 2);
    float cse_P_22 = cse_P_20*cse_P_21;
    float cse_P_23 = 2*cse_P_2;
    float cse_P_24 = 2*S_inv[3][4]*cse_P_16 + 2*S_inv[4][4]*cse_P_12 - 2*cse_P_3*(P[0][0]*S_inv[0][4] + P[0][1]*S_inv[1][4] + S_inv[2][4]*cse_P_0);
    float cse_P_25 = S_inv[3][3]*cse_P_16 + S_inv[4][3]*cse_P_12 - cse_P_3*(P[0][0]*S_inv[0][3] + P[0][1]*S_inv[1][3] + S_inv[2][3]*cse_P_0);
    float cse_P_26 = -cse_P_14*cse_P_25 + 8*cse_P_19*cse_P_22 + cse_P_23*(S_inv[3][0]*cse_P_16 + S_inv[4][0]*cse_P_12 - cse_P_3*(P[0][0]*S_inv[0][0] + P[0][1]*S_inv[1][0] + S_inv[2][0]*cse_P_0)) + cse_P_24*cse_P_6;
    float cse_P_27 = 4*cse_P_9;
    float cse_P_28 = EKF_WHEEL_BASE*(cse_P_10*cse_P_24 + cse_P_15*cse_P_25 - cse_P_27*(S_inv[3][1]*cse_P_16 + S_inv[4][1]*cse_P_12 + cse_P_17 - cse_P_3*(P[0][0]*S_inv[0][1] + P[0][1]*S_inv[1][1] + S_inv[2][1]*cse_P_0)));
    float cse_P_29 = (1.0/16.0)/(cse_P_19*cse_P_20*cse_P_21);
    float cse_P_30 = 2*cse_P_26;
    float cse_P_31 = P[1][1] + P[1][2];
    float cse_P_32 = 2*P[1][0];
    float cse_P_33 = EKF_WHEEL_BASE*P[1][1];
    float cse_P_34 = 2*cse_P_10*cse_P_33 - 2*cse_P_32*cse_P_6;
    float cse_P_35 = cse_P_14*cse_P_32 + cse_P_15*cse_P_33;
    float cse_P_36 = S_inv[3][2]*cse_P_35 + S_inv[4][2]*cse_P_34 - cse_P_3*(P[1][0]*S_inv[0][2] + P[1][1]*S_inv[1][2] + S_inv[2][2]*cse_P_31);
    float cse_P_37 = cse_P_3*cse_P_36;
    float cse_P_38 = 2*S_inv[3][4]*cse_P_35 + 2*S_inv[4][4]*cse_P_34 - 2*cse_P_3*(P[1][0]*S_inv[0][4] + P[1][1]*S_inv[1][4] + S_inv[2][4]*cse_P_31);
    float cse_P_39 = S_inv[3][3]*cse_P_35 + S_inv[4][3]*cse_P_34 - cse_P_3*(P[1][0]*S_inv[0][3] + P[1][1]*S_inv[1][3] + S_inv[2][3]*cse_P_31);
    float cse_P_40 = -cse_P_14*cse_P_39 + cse_P_23*(S_inv[3][0]*cse_P_35 + S_inv[4][0]*cse_P_34 - cse_P_3*(P[1][0]*S_inv[0][0] + P[1][1]*S_inv[1][0] + S_inv[2][0]*cse_P_31)) + cse_P_38*cse_P_6;
    float cse_P_41 = 16*EKF_WHEEL_BASE*cse_P_22 - cse_P_10*cse_P_38 - cse_P_15*cse_P_39 + cse_P_27*(S_inv[3][1]*cse_P_35 + S_inv[4][1]*cse_P_34 - cse_P_3*(P[1][0]*S_inv[0][1] + P[1][1]*S_inv[1][1] + S_inv[2][1]*cse_P_31) + cse_P_36);
    float cse_P_42 = EKF_WHEEL_BASE*cse_P_41;
    float cse_P_43 = 2*cse_P_40;
    float cse_P_44 = 2*P[2][0];
    float cse_P_45 = EKF_WHEEL_BASE*P[2][1];
    float cse_P_46 = cse_P_14*cse_P_44 + cse_P_15*cse_P_45;
    float cse_P_47 = 2*cse_P_10*cse_P_45 - 2*cse_P_44*cse_P_6;
    float cse_P_48 = P[2][1] + P[2][2];
    float cse_P_49 = S_inv[3][2]*cse_P_46 + S_inv[4][2]*cse_P_47 - cse_P_3*(P[2][0]*S_inv[0][2] + P[2][1]*S_inv[1][2] + S_inv[2][2]*cse_P_48);
    float cse_P_50 = cse_P_3 + cse_P_49;
    float cse_P_51 = 2*S_inv[3][4]*cse_P_46 + 2*S_inv[4][4]*cse_P_47 - 2*cse_P_3*(P[2][0]*S_inv[0][4] + P[2][1]*S_inv[1][4] + S_inv[2][4]*cse_P_48);
    float cse_P_52 = S_inv[3][3]*cse_P_46 + S_inv[4][3]*cse_P_47 - cse_P_3*(P[2][0]*S_inv[0][3] + P[2][1]*S_inv[1][3] + S_inv[2][3]*cse_P_48);
    float cse_P_53 = cse_P_14*cse_P_52 - cse_P_23*(S_inv[3][0]*cse_P_46 + S_inv[4][0]*cse_P_47 - cse_P_3*(P[2][0]*S_inv[0][0] + P[2][1]*S_inv[1][0] + S_inv[2][0]*cse_P_48)) - cse_P_51*cse_P_6;
    float cse_P_54 = cse_P_10*cse_P_51 + cse_P_15*cse_P_52 - cse_P_27*(S_inv[3][1]*cse_P_46 + S_inv[4][1]*cse_P_47 - cse_P_3*(P[2][0]*S_inv[0][1] + P[2][1]*S_inv[1][1] + S_inv[2][1]*cse_P_48) + cse_P_49);
    float cse_P_55 = EKF_WHEEL_BASE*cse_P_54;
    float cse_P_56 = 2*cse_P_53;
    float next_P_0_0 = cse_P_29*(-P[1][0]*cse_P_28 + P[2][0]*cse_P_18 + cse_P_26*cse_P_7);
    float next_P_0_1 = cse_P_29*(P[0][1]*cse_P_30 - P[1][1]*cse_P_28 + P[2][1]*cse_P_18);
    float next_P_0_2 = cse_P_29*(P[0][2]*cse_P_30 - P[1][2]*cse_P_28 + P[2][2]*cse_P_18);
    float next_P_1_0 = cse_P_29*(P[1][0]*cse_P_42 + P[2][0]*cse_P_37 + cse_P_40*cse_P_7);
    float next_P_1_1 = cse_P_29*(P[0][1]*cse_P_43 + P[2][1]*cse_P_37 + cse_P_33*cse_P_41);
    float next_P_1_2 = cse_P_29*(P[0][2]*cse_P_43 + P[1][2]*cse_P_42 + P[2][2]*cse_P_37);
    float next_P_2_0 = cse_P_29*(4*EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*P[2][0]*cse_P_50 - P[1][0]*cse_P_55 - cse_P_53*cse_P_7);
    float next_P_2_1 = cse_P_29*(4*EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*P[2][1]*cse_P_50 - P[0][1]*cse_P_56 - cse_P_33*cse_P_54);
    float next_P_2_2 = cse_P_29*(4*EKF_TAU_L*EKF_TAU_R*EKF_WHEEL_BASE*P[2][2]*cse_P_50 - P[0][2]*cse_P_56 - P[1][2]*cse_P_55);
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
