// ==========================================
// ekf generated defines
// ==========================================
#define EKF_STATE_DIM 3
#define EKF_MEASURE_DIM 5
#define EKF_CONTROL_DIM 2
#define EKF_WHEEL_BASE 0.30f
#define EKF_K_R 1.0f
#define EKF_K_L 1.0f
#define EKF_TAU_R 0.15f
#define EKF_TAU_L 0.15f
#define EKF_IMU_RX 0.1f
#define EKF_IMU_RY 0.1f
// ==========================================
// end ekf generated defines
// ==========================================

#ifndef TINYEKF_H
#define TINYEKF_H

#include <cstdint>

// **********************************
// *       Class of TinyEKF         *
// **********************************

class TinyEKF {
public:
    TinyEKF() {}

    void init(const float x0[EKF_STATE_DIM], 
              const float P0[EKF_STATE_DIM][EKF_STATE_DIM], 
              const float Q_init[EKF_STATE_DIM][EKF_STATE_DIM], 
              const float R_init[EKF_MEASURE_DIM][EKF_MEASURE_DIM]) {
        for(uint8_t i = 0; i < EKF_STATE_DIM; i++) {
            x[i] = x0[i];
            for(uint8_t j = 0; j < EKF_STATE_DIM; j++) {
                P[i][j] = P0[i][j];
                Q[i][j] = Q_init[i][j];
            }
        }
        for(uint8_t i = 0; i < EKF_MEASURE_DIM; i++)
            for(uint8_t j = 0; j < EKF_MEASURE_DIM; j++)
                R[i][j] = R_init[i][j];
    }

    void predict(const float u[EKF_CONTROL_DIM]);
    void update(const float z[EKF_MEASURE_DIM], const float u[EKF_CONTROL_DIM]);
    float get_state(uint8_t index) const { return x[index]; }
private:
    float x[EKF_STATE_DIM];
    float P[EKF_STATE_DIM][EKF_STATE_DIM];  
    float Q[EKF_STATE_DIM][EKF_STATE_DIM];
    float R[EKF_MEASURE_DIM][EKF_MEASURE_DIM];
};

#endif // TINYEKF_H