#ifndef EKF_H
#define EKF_H

#include "Eigen/Dense"

#include "constants.h"

class EKF
{
public:
    EKF();
    ~EKF();

private:
    Eigen::Vector<int, NUM_STATES> x; // we define state as x,y,z,vx,vy,vz,ax,ay,az. x is defined as up
    Eigen::Vector<int, NUM_STATES> x_priori;
    Eigen::Matrix<int, NUM_STATES, NUM_STATES> P; // covariance matrix
    Eigen::Matrix<int, NUM_STATES, NUM_STATES> P_priori;
    Eigen::Matrix<int, NUM_STATES, NUM_STATES> Q;       // process noise matrix
    Eigen::Matrix<int, NUM_MEASUREMENTS, NUM_STATES> H; // observation matrix. we are only tracking accelerometer measurements for now
    Eigen::Matrix<int, NUM_STATES, NUM_MEASUREMENTS> K; // Kalman gain matrix
};

#endif // EKF_H
