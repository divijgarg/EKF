#include "EKF.h"

#include <iostream>
#include <cmath>
#include "Eigen/Dense"

EKF::EKF()
{
    init();
}

void EKF::init() 
{
    x_est.setZero();
    x_pred.setZero();
    z.setZero();
    u.setZero();

    P_est = Eigen::Matrix<double, NUM_STATES, NUM_STATES>::Identity() * 1e1;
    P_pred = Eigen::Matrix<double, NUM_STATES, NUM_STATES>::Identity() * 1e1;

    setQ(set_dt);
    setR();
    setH();
    setF(set_dt);
    setG(set_dt);
    K.setZero();
    id.setIdentity();
    predict();
}

void EKF::predict()
{
    x_pred = F * x_est + G * u;
    P_pred = F * P_est * F.transpose() + Q;
}

void EKF::compute_K()
{
    K = P_pred * H.transpose() * (H * P_pred * H.transpose() + R).inverse();
}

void EKF::update()
{
    x_est = x_pred + K * (z - H * x_pred);
    P_est = (id - K * H) * P_pred * (id - K * H).transpose() + K * R * K.transpose();
}

void EKF::tick() {
    compute_K();
    update();
    predict();
    printMatrix("x_est: ", x_est);
}

void EKF::setR() {
    R.setZero();
    R(0, 0) = sigma_x * sigma_x;
    R(1, 1) = sigma_y * sigma_y;
    R(2, 2) = sigma_z * sigma_z;
}

void EKF::setQ(double dt)
{
    Q.setZero();
    Q(0, 0) = 0.25 * dt * dt * dt * dt;
    Q(0, 1) = 0.5 * dt * dt * dt;
    Q(1, 0) = 0.5 * dt * dt * dt;
    Q(1, 1) = dt * dt;

    Q(2, 2) = 0.25 * dt * dt * dt * dt;
    Q(2, 3) = 0.5 * dt * dt * dt;
    Q(3, 2) = 0.5 * dt * dt * dt;
    Q(3, 3) = dt * dt;

    Q(4, 4) = 0.25 * dt * dt * dt * dt;
    Q(4, 5) = 0.5 * dt * dt * dt;
    Q(5, 4) = 0.5 * dt * dt * dt;
    Q(5, 5) = dt * dt;

    Q = Q * sigma_a * sigma_a;
}
void EKF::setF(double dt)
{
    F.setZero();
    F(0, 0) = 1;
    F(0, 1) = dt;
    F(1, 1) = 1;

    F(2, 2) = 1;
    F(2, 3) = dt;
    F(3, 3) = 1;

    F(4, 4) = 1;
    F(4, 5) = dt;
    F(5, 5) = 1;
}
void EKF::setG(double dt)
{
    G.setZero();
    G(0, 0) = 0.5 * dt * dt;
    G(1, 0) = dt;
    G(2, 1) = 0.5 * dt * dt;
    G(3, 1) = dt;
    G(4, 2) = 0.5 * dt * dt;
    G(5, 2) = dt;
    printMatrix("G: ", G);
}
void EKF::setH()
{
    H.setZero();
    H(0, 0) = 1;
    H(1, 2) = 1;
    H(2, 4) = 1;
}

const Eigen::Matrix<double, NUM_STATES, 1> &EKF::get_x()
{
    return x_est;
}

void EKF::set_measurement(Eigen::Matrix<double, NUM_MEASUREMENTS, 1> input)
{
    z = input;
}

void EKF::set_control(Eigen::Matrix<double, NUM_CONTROLS, 1> input)
{
    u = input;
}

EKF::~EKF() {}
