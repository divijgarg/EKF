#ifndef EKF_H
#define EKF_H

#include <iostream>
#include <string>

#include "Eigen/Dense"

#include "constants.h"

template <typename Derived>
void printMatrix(const std::string &name, const Eigen::MatrixBase<Derived> &m)
{
    std::cout << name << " (" << m.rows() << "x" << m.cols() << "):\n"
              << m << "\n"
              << std::endl;
}

class EKF
{
public:
    EKF();
    ~EKF();
    void init(); // initializes the entire filter
    void setF(double dt);
    void setG(double dt);
    void setH();
    void setR();
    void setQ(double dt);
    void compute_K(); // calculate the kalman gain
    void tick();

    void update();
    void predict();

private: 
    Eigen::Matrix<double, NUM_STATES, 1> x_est; //estimate of i at time step i
    Eigen::Matrix<double, NUM_STATES, 1> x_pred; //prediction of i+1 at time step i
    Eigen::Matrix<double, NUM_STATES, NUM_STATES> P_est; // estimate of i at time step i
    Eigen::Matrix<double, NUM_STATES, NUM_STATES> P_pred; //prediction of i+1 at time step i

    Eigen::Matrix<double, NUM_MEASUREMENTS, 1> z; // assume we're getting positional data only
    Eigen::Matrix<double, NUM_CONTROLS, 1> u;     // our control input is acceleration
    
    Eigen::Matrix<double, NUM_STATES, NUM_STATES> Q;             // process noise matrix
    Eigen::Matrix<double, NUM_MEASUREMENTS, NUM_STATES> H;       // observation matrix. we are only tracking accelerometer measurements for now
    Eigen::Matrix<double, NUM_STATES, NUM_MEASUREMENTS> K;       // Kalman gain matrix
    Eigen::Matrix<double, NUM_MEASUREMENTS, NUM_MEASUREMENTS> R; // Measurement noise covariance matrix
    Eigen::Matrix<double, NUM_STATES, NUM_STATES> F;             // State transition matrix
    Eigen::Matrix<double, NUM_STATES, NUM_CONTROLS> G;           // State transition matrix
    Eigen::Matrix<double, NUM_STATES, NUM_STATES> id;
};

#endif