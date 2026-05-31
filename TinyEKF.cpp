#include "TinyEKF.h"
#include <cmath>

void TinyEKF::predict(const float u[]) {
    // calculating x
    float next_x_0 = 0.5*u[0]*cos(x[0]) + x[0] + 0.01*x[1];
    float next_x_1 = x[1] + 0.01*sin(x[0]);
    // update x in memory
    x[0] = next_x_0;
    x[1] = next_x_1;
    // calculating Q
    float x0 = -0.5*u[0]*sin(x[0]) + 1;
    float x1 = P[0][1]*x0;
    float x2 = P[0][0]*x0 + 0.01*P[1][0];
    float x3 = 0.01*P[1][1];
    float x4 = cos(x[0]);
    float x5 = 0.01*x4;
    float x6 = P[0][0]*x5 + P[1][0];
    float next_Q_0_0 = 0.0001*P[1][1] + Q[0][0] + x0*x2 + 0.01*x1;
    float next_Q_0_1 = Q[0][1] + x1 + x2*x5 + x3;
    float next_Q_1_0 = 0.0001*P[0][1]*x4 + Q[1][0] + x0*x6 + x3;
    float next_Q_1_1 = P[0][1]*x5 + P[1][1] + Q[1][1] + x5*x6;
    // update Q in memory
    Q[0][0] = next_Q_0_0;
    Q[0][1] = next_Q_0_1;
    Q[1][0] = next_Q_1_0;
    Q[1][1] = next_Q_1_1;
}

void TinyEKF::update(const float z[]) {
    // calculating x
    float x0 = pow(x[0], 2);
    float x1 = pow(x[1], 2);
    float x2 = sqrt(x0 + x1);
    float x3 = -x2;
    float x4 = x3 + z[0];
    float x5 = R[1][1]*x0;
    float x6 = R[1][1]*x1;
    float x7 = P[0][0]*x0;
    float x8 = P[1][1]*x1;
    float x9 = P[0][1]*x[1];
    float x10 = x9*x[0];
    float x11 = P[1][0]*x[0]*x[1];
    float x12 = x10 + x11 + x7 + x8;
    float x13 = x12 + x5 + x6;
    float x14 = R[1][0]*x0;
    float x15 = R[1][0]*x1;
    float x16 = 1.0/(R[0][0]*x10 + R[0][0]*x11 + R[0][0]*x5 + R[0][0]*x6 + R[0][0]*x7 + R[0][0]*x8 - R[0][1]*x10 - R[0][1]*x11 - R[0][1]*x14 - R[0][1]*x15 - R[0][1]*x7 - R[0][1]*x8 - R[1][0]*x10 - R[1][0]*x11 - R[1][0]*x7 - R[1][0]*x8 + R[1][1]*x10 + R[1][1]*x11 + R[1][1]*x7 + R[1][1]*x8);
    float x17 = 1.0/x2;
    float x18 = x17*x[0];
    float x19 = x16*(P[0][0]*x18 + x17*x9);
    float x20 = -x12 - x14 - x15;
    float x21 = x3 + z[1];
    float x22 = R[0][0]*x0 + R[0][0]*x1 + x12;
    float x23 = -R[0][1]*x0 - R[0][1]*x1 - x12;
    float x24 = x16*(P[1][0]*x18 + P[1][1]*x17*x[1]);
    float next_x_0 = x21*(x19*x22 + x19*x23) + x4*(x13*x19 + x19*x20) + x[0];
    float next_x_1 = x21*(x22*x24 + x23*x24) + x4*(x13*x24 + x20*x24) + x[1];
    // update x in memory
    x[0] = next_x_0;
    x[1] = next_x_1;
    // calculating P
    float x0 = pow(x[0], 2);
    float x1 = R[0][0]*x0;
    float x2 = pow(x[1], 2);
    float x3 = R[0][0]*x2;
    float x4 = P[0][0]*x0;
    float x5 = P[1][1]*x2;
    float x6 = P[0][1]*x[1];
    float x7 = x6*x[0];
    float x8 = P[1][0]*x[0]*x[1];
    float x9 = x4 + x5 + x7 + x8;
    float x10 = x1 + x3 + x9;
    float x11 = R[0][1]*x0;
    float x12 = R[0][1]*x2;
    float x13 = 1.0/(R[0][0]*x4 + R[0][0]*x5 + R[0][0]*x7 + R[0][0]*x8 - R[0][1]*x4 - R[0][1]*x5 - R[0][1]*x7 - R[0][1]*x8 - R[1][0]*x11 - R[1][0]*x12 - R[1][0]*x4 - R[1][0]*x5 - R[1][0]*x7 - R[1][0]*x8 + R[1][1]*x1 + R[1][1]*x3 + R[1][1]*x4 + R[1][1]*x5 + R[1][1]*x7 + R[1][1]*x8);
    float x14 = pow(x0 + x2, -1.0/2.0);
    float x15 = x14*x[0];
    float x16 = x13*(P[0][0]*x15 + x14*x6);
    float x17 = -x11 - x12 - x9;
    float x18 = x10*x16 + x16*x17;
    float x19 = x14*x[1];
    float x20 = R[1][1]*x0 + R[1][1]*x2 + x9;
    float x21 = -R[1][0]*x0 - R[1][0]*x2 - x9;
    float x22 = x16*x20 + x16*x21;
    float x23 = -x18*x19 - x19*x22;
    float x24 = -x15*x18 - x15*x22 + 1;
    float x25 = x13*(P[1][0]*x15 + P[1][1]*x19);
    float x26 = x10*x25 + x17*x25;
    float x27 = x20*x25 + x21*x25;
    float x28 = -x15*x26 - x15*x27;
    float x29 = -x19*x26 - x19*x27 + 1;
    float next_P_0_0 = P[0][0]*x24 + P[1][0]*x23;
    float next_P_0_1 = P[0][1]*x24 + P[1][1]*x23;
    float next_P_1_0 = P[0][0]*x28 + P[1][0]*x29;
    float next_P_1_1 = P[0][1]*x28 + P[1][1]*x29;
    // update P in memory
    P[0][0] = next_P_0_0;
    P[0][1] = next_P_0_1;
    P[1][0] = next_P_1_0;
    P[1][1] = next_P_1_1;
}
