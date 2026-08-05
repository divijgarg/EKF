#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "Eigen/Dense"

constexpr int NUM_STATES = 6;       // x,vx,y,vy,z,vz
constexpr int NUM_CONTROLS =0; //ax,ay,az
constexpr int NUM_MEASUREMENTS = 3; // x,y,z
constexpr double set_dt = 10;  //seconds


constexpr double sigma_v = 7.615435e-05 
constexpr double sigma_u =9.401772e-13
constexpr double sigma_mag = 0.3
constexpr double sigma_sun = 0.000872663


#endif
