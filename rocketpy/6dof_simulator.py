from rocketpy import Environment, SolidMotor, Rocket, Flight

import datetime
import math

import numpy as np
import numpy.linalg as la




tomorrow = datetime.date.today() + datetime.timedelta(days=1)



# 1 · Describe the sky — real forecast weather
env = Environment(latitude=32.99, longitude=-106.97, elevation=1400)
env.set_date(
    (tomorrow.year, tomorrow.month, tomorrow.day, 12)
)  # Hour given in UTC time

env.set_atmospheric_model(type="Forecast", file="GFS")

# 2 · Build the motor (SolidMotor here — also Hybrid/Liquid/Generic)
motor = SolidMotor(
    thrust_source="Cesaroni_M1670.eng",
    dry_mass=1.815,
    dry_inertia=(0.125, 0.125, 0.002),
    nozzle_radius=33 / 1000,
    grain_number=5,
    grain_density=1815,
    grain_outer_radius=33 / 1000,
    grain_initial_inner_radius=15 / 1000,
    grain_initial_height=120 / 1000,
    grain_separation=5 / 1000,
    grains_center_of_mass_position=0.397,
    center_of_dry_mass_position=0.317,
    nozzle_position=0,
    burn_time=3.9,
    throat_radius=11 / 1000,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)
# 3 · Assemble the rocket
calisto = Rocket(
    radius=127 / 2000,
    mass=14.426,
    inertia=(6.321, 6.321, 0.034),
    power_off_drag="powerOffDragCurve.csv",
    power_on_drag="powerOnDragCurve.csv",
    center_of_mass_without_motor=0,
    coordinate_system_orientation="tail_to_nose",
)
rail_buttons = calisto.set_rail_buttons(
    upper_button_position=0.0818,
    lower_button_position=-0.618,
    angular_position=45,
)
calisto.add_motor(motor, position=-1.255)
nose_cone = calisto.add_nose(length=0.55829, kind="vonKarman", position=1.278)

fin_set = calisto.add_trapezoidal_fins(
    n=4,
    root_chord=0.120,
    tip_chord=0.060,
    span=0.110,
    position=-1.04956,
    cant_angle=0.5,
    airfoil=("NACA0012-radians.txt", "radians"),
)

tail = calisto.add_tail(
    top_radius=0.0635, bottom_radius=0.0435, length=0.060, position=-1.194656
)
Main = calisto.add_parachute(
    "Main",
    cd_s=10.0,
    trigger=800,
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
)

Drogue = calisto.add_parachute(
    "Drogue",
    cd_s=1.0,
    trigger="apogee",
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
)

# 4 · Fly it — full 6-DOF simulation
flight = Flight(
    rocket=calisto, environment=env, rail_length=5.2, inclination=85, heading=0
)



# 5 · Log the flight on a uniform grid in the same layout as
# mekf_python/test_data_generator.py:
#   t, y1 (magnetometer), y2 (accelerometer), w (gyro), q_true
# followed by the raw inertial position, velocity and acceleration. y1/y2 are
# noisy body-frame unit vectors, so the magnetometer and the accelerometer each
# appear twice: once as a filter measurement, once as raw ENU state.
# RocketPy is scalar-first (e0 is the scalar), so the quaternion is written
# scalar-last to match the filter.
G = 9.80665          # gravity used to turn inertial accel into specific force


def log_flight(flight, path="flight_data.csv", time_step=0.01,
               meas_noise=0.01, sigma_v=0.01,
               b_true=(0.005, -0.003, 0.002), seed=23):
    rng = np.random.default_rng(seed)
    b_true = np.asarray(b_true, dtype=float)

    # values below are approximate for 32.99, -106.97
    DECLINATION = math.radians(8.0)    # positive east of true north
    INCLINATION = math.radians(58.5)   # positive downward

    MAG_REF = np.array([
        math.cos(INCLINATION) * math.sin(DECLINATION),   # east
        math.cos(INCLINATION) * math.cos(DECLINATION),   # north
        -math.sin(INCLINATION),                          # up
    ])

    # attitude matrix from a scalar-last quaternion (same convention as
    # attitude_functions.return_A)
    def return_A(q):
        q_vec = q[0:3]
        cross = np.array([
            [        0, -q_vec[2],  q_vec[1]],
            [ q_vec[2],         0, -q_vec[0]],
            [-q_vec[1],  q_vec[0],         0],
        ])
        return ((2 * q[3] ** 2 - 1) * np.identity(3)
                - 2 * q[3] * cross
                + 2 * np.outer(q_vec, q_vec))

    rows = []
    n_steps = int(flight.t_final / time_step) + 1
    for i in range(n_steps):
        t = i * time_step

        q_true = np.array([flight.e1(t), flight.e2(t), flight.e3(t), flight.e0(t)])
        A_t = return_A(q_true)

        pos = np.array([flight.x(t), flight.y(t), flight.z(t)])
        vel = np.array([flight.vx(t), flight.vy(t), flight.vz(t)])
        acc = np.array([flight.ax(t), flight.ay(t), flight.az(t)])

        # accelerometer senses specific force a - g, i.e. a + g*up in ENU
        f_ref = acc + np.array([0.0, 0.0, G])
        norm = la.norm(f_ref)
        r2 = f_ref / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])

        y1 = A_t @ MAG_REF + rng.normal(0, meas_noise, 3)
        y2 = A_t @ r2 + rng.normal(0, meas_noise, 3)
        w_meas = (np.array([flight.w1(t), flight.w2(t), flight.w3(t)])
                  + b_true + rng.normal(0, sigma_v / np.sqrt(time_step), 3))

        rows.append(np.concatenate(([t], y1, y2, w_meas, q_true, pos, vel, acc)))

    header = ("t, y1_x, y1_y, y1_z, y2_x, y2_y, y2_z, w_x, w_y, w_z, "
              "q_true_1, q_true_2, q_true_3, q_true_4, "
              "x, y, z, vx, vy, vz, ax, ay, az   "
              f"| dt={time_step}  y1=magnetometer r1={np.round(MAG_REF, 6).tolist()}  "
              "y2=accelerometer r2=unit(accel+[0,0,g])  "
              f"true bias={b_true.tolist()}  meas_noise={meas_noise}  sigma_v={sigma_v}  "
              f"g={G}  frame=ENU (m, m/s, m/s^2)  q scalar-last")

    np.savetxt(path, np.array(rows), delimiter=",", header=header, fmt="%.8f")

    print(f"done writing {path}: t = 0 to {flight.t_final:.2f} s at {time_step} s steps")


log_flight(flight)


flight.all_info()


# Animate the full trajectory â€” rocket moves through 3D space
# Press Escape or close the window to exit the animation
flight.plots.animate_trajectory(
    start=0,
    stop=flight.t_final,
    time_step=0.05,
    color_by="speed",
    show_kinematic_plots=True,
    camera_mode="follow",
)

# Alternatively, animate attitude and stability diagnostics
# test_flight.plots.animate_rotate(
#     start=0,
#     stop=test_flight.t_final,
#     time_step=0.05,
#     show_attitude_plots=True,
#     show_cp_cm=True,
# )

# Deterministic export:
# test_flight.plots.animate_trajectory(export_file="flight.mp4", export_fps=30)

# To use your own 3D model, pass its path via file_name:
# test_flight.plots.animate_trajectory(file_name="my_rocket.stl")