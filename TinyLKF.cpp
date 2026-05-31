#include "TinyLKF.h"

void TinyLKF::predict(const float u[]) {    // --- calculating x ---
    float next_x_0 = 5.0000000000000002e-5*u[0] + x[0] + 0.01*x[1];
    float next_x_1 = 0.01*u[0] + x[1];
    // updating x in memory
    x[0] = next_x_0;
    x[1] = next_x_1;
    // --- calculating P ---
    float next_P_0_0 = P[0][0] + 0.01*P[0][1] + 0.01*P[1][0] + 0.0001*P[1][1] + Q[0][0];
    float next_P_0_1 = P[0][1] + 0.01*P[1][1] + Q[0][1];
    float next_P_1_0 = P[1][0] + 0.01*P[1][1] + Q[1][0];
    float next_P_1_1 = P[1][1] + Q[1][1];
    // updating P in memory
    P[0][0] = next_P_0_0;
    P[0][1] = next_P_0_1;
    P[1][0] = next_P_1_0;
    P[1][1] = next_P_1_1;
}
void TinyLKF::update(const float z[]) {    // --- calculating x ---
    float next_x_0 = (P[0][0]*z[0] + R[0][0]*x[0])/(P[0][0] + R[0][0]);
    float next_x_1 = (-P[1][0]*(x[0] - z[0]) + x[1]*(P[0][0] + R[0][0]))/(P[0][0] + R[0][0]);
    // updating x in memory
    x[0] = next_x_0;
    x[1] = next_x_1;
    // --- calculating P ---
    float next_P_0_0 = P[0][0]*R[0][0]/(P[0][0] + R[0][0]);
    float next_P_0_1 = P[0][1]*R[0][0]/(P[0][0] + R[0][0]);
    float next_P_1_0 = P[1][0]*R[0][0]/(P[0][0] + R[0][0]);
    float next_P_1_1 = (-P[0][1]*P[1][0] + P[1][1]*(P[0][0] + R[0][0]))/(P[0][0] + R[0][0]);
    // updating P in memory
    P[0][0] = next_P_0_0;
    P[0][1] = next_P_0_1;
    P[1][0] = next_P_1_0;
    P[1][1] = next_P_1_1;
}
