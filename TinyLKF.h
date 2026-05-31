#ifndef TINYLKF_H
#define TINYLKF_H

#include <cstdint>

// **********************************
// *       Class of TinyLKF         *
// **********************************

#define LKF_STATE_DIM 2    
#define LKF_MEASURE_DIM 1  
#define LKF_CONTROL_DIM 1 

class TinyLKF {
public:
    TinyLKF() {}

    void init(const float x0[LKF_STATE_DIM], 
              const float P0[LKF_STATE_DIM][LKF_STATE_DIM], 
              const float Q_init[LKF_STATE_DIM][LKF_STATE_DIM], 
              const float R_init[LKF_MEASURE_DIM][LKF_MEASURE_DIM]) {
        for(uint8_t i = 0; i < LKF_STATE_DIM; i++) {
            x[i] = x0[i];
            for(uint8_t j = 0; j < LKF_STATE_DIM; j++) {
                P[i][j] = P0[i][j];
                Q[i][j] = Q_init[i][j];
            }
        }
        for(uint8_t i = 0; i < LKF_MEASURE_DIM; i++)
            for(uint8_t j = 0; j < LKF_MEASURE_DIM; j++)
                R[i][j] = R_init[i][j];
    }

    void predict(const float u[LKF_CONTROL_DIM]);
    void update(const float z[LKF_MEASURE_DIM]);
    float get_state(uint8_t index) const { return x[index]; }
private:
    float x[LKF_STATE_DIM];
    float P[LKF_STATE_DIM][LKF_STATE_DIM];  
    float Q[LKF_STATE_DIM][LKF_STATE_DIM];
    float R[LKF_MEASURE_DIM][LKF_MEASURE_DIM];
};

#endif // TINYLKF_H