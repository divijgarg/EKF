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

    P = Eigen::Matrix<double, NUM_STATES, NUM_STATES>::Identity() * 1e1;
    P(0, 0) = sigma_v * *2;
    P(1, 1) = sigma_v * *2;
    P(2, 2) = sigma_v * *2;
    P(3, 3) = sigma_u * *2;
    P(4, 4) = sigma_u * *2;
    P(5, 5) = sigma_u * *2;

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

void EKF::tick()
{
    compute_K();
    update();
    predict();
}

void EKF::setR()
{
    R.setZero();
    R.block<3, 3>(0, 0) = sigma_sun * sigma_sun * Eigen::Matrix3d::Identity();
    R.block<3, 3>(3, 3) = sigma_mag * sigma_mag * Eigen::Matrix3d::Identity();
}

void EKF::setQ(double dt)
{
    Q.setZero();
    Q.block<3, 3>(0, 0) = sigma_v * sigma_v * dt + 0.5 * sigma_u * sigma_u * dt * dt * dt * Eigen::Matrix3d::Identity();
    Q.block<3, 3>(0, 3) = -0.5 * sigma_u * sigma_u * dt * dt;
    *Eigen::Matrix3d::Identity();
    Q.block<3, 3>(3, 0) = -0.5 * sigma_u * sigma_u * dt * dt;
    *Eigen::Matrix3d::Identity();
    Q.block<3, 3>(3, 3) = sigma_u * sigma_u * dt;
    *Eigen::Matrix3d::Identity();
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
    G.block<3, 3>(0, 0) = -1.0 * Eigen::Matrix3d::Identity();
    G.block<3, 3>(3, 3) = -1.0 * Eigen::Matrix3d::Identity();
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
