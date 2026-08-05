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
    void init(); 
    void tick();

    const Eigen::Matrix<double, NUM_STATES, 1> &get_x();
    void set_measurement(Eigen::Matrix<double,NUM_MEASUREMENTS,1> input);
    void set_control(Eigen::Matrix<double,NUM_CONTROLS,1> input);



private:

    void setF(double dt);
    void setG(double dt);
    void setH();
    void setR();
    void setQ(double dt);
    void compute_K(); // calculate the kalman gain
    void update();
    void predict();
    
    Eigen::Matrix<double, NUM_STATES, 1> x;           
    Eigen::Matrix<double, NUM_STATES, NUM_STATES> P;  
    Eigen::Matrix<double, NUM_MEASUREMENTS, 1> z; // gyro data only
    Eigen::Matrix<double, NUM_CONTROLS, 1> u;     // no control!
    Eigen::Matrix<double, NUM_STATES, NUM_STATES> Q;             // process noise matrix
    Eigen::Matrix<double, NUM_MEASUREMENTS, NUM_STATES> H;       // observation matrix. 
    Eigen::Matrix<double, NUM_STATES, NUM_MEASUREMENTS> K;       // Kalman gain matrix
    Eigen::Matrix<double, NUM_MEASUREMENTS, NUM_MEASUREMENTS> R; // Measurement noise covariance matrix
    Eigen::Matrix<double, NUM_STATES, NUM_STATES> F;             // State transition matrix
    Eigen::Matrix<double, NUM_STATES, NUM_CONTROLS> G;           // State transition matrix
    Eigen::Matrix<double, NUM_STATES, NUM_STATES> id;

};

#endif