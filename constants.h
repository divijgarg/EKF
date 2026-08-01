#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "Eigen/Dense"

constexpr int NUM_STATES = 6;       // x,vx,y,vy,z,vz
constexpr int NUM_CONTROLS =3; //ax,ay,az
constexpr int NUM_MEASUREMENTS = 3; // x,y,z
constexpr double set_dt = 0.05;  //seconds

//errors

constexpr double sigma_a = 0.2; //random accel standard deviation, m/s^2

constexpr double sigma_x = 3; //x measurement standard deviation, m
constexpr double sigma_y = 3;//y measurement standard deviation, m
constexpr double sigma_z = 3;//z measurement standard deviation, m

#endif
