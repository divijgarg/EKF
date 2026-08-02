#include "EKF.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>


#include "Eigen/Dense"

constexpr int NUM_TICKS = 60000;

const std::string DATA_DIR =
    "Flight Data/CATS/cats_euroc23_full_datasets/decoded/01_astg__main/";


static std::vector<std::string> split_csv_line(const std::string &line)
{
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ','))
    {
        fields.push_back(field);
    }
    return fields;
}

static bool read_csv_columns(const std::string &path,
                             const std::vector<int> &columns,
                             const std::vector<std::vector<double> *> &out)
{
    std::ifstream in(path);
    if (!in)
    {
        std::cerr << "Failed to open " << path << std::endl;
        return false;
    }

    std::string line;
    std::getline(in, line); // header

    while (std::getline(in, line))
    {
        if (line.empty())
            continue;

        const std::vector<std::string> fields = split_csv_line(line);

        std::vector<double> values(columns.size());
        bool ok = true;
        for (size_t i = 0; i < columns.size() && ok; ++i)
        {
            const int c = columns[i];
            if (c >= static_cast<int>(fields.size()))
            {
                ok = false;
                break;
            }
            try
            {
                values[i] = std::stod(fields[c]);
            }
            catch (const std::exception &)
            {
                ok = false;
            }
        }
        if (!ok)
            continue;

        for (size_t i = 0; i < values.size(); ++i)
        {
            out[i]->push_back(values[i]);
        }
    }
    return true;
}

int main()
{
    EKF ekf;

    // imu.csv:                    t_s, id, Ax_mps2, Ay_mps2, Az_mps2, Gx, Gy, Gz
    // baro_altitude_filtered.csv: t_s, filtered_altitude_AGL, filtered_acceleration
    std::vector<double> time;        // s
    std::vector<double> accel_x;     // m/s^2
    std::vector<double> accel_y;     // m/s^2
    std::vector<double> accel_z;     // m/s^2
    std::vector<double> baro_height; // m AGL

    if (!read_csv_columns(DATA_DIR + "imu.csv", {0, 2, 3, 4},
                          {&time, &accel_x, &accel_y, &accel_z}))
    {
        return 1;
    }
    if (!read_csv_columns(DATA_DIR + "baro_altitude_filtered.csv", {1},
                          {&baro_height}))
    {
        return 1;
    }

    std::cout << "IMU samples:  " << accel_x.size() << "\n"
              << "Baro samples: " << baro_height.size() << "\n"
              << "t: " << time.front() << " s .. " << time.back() << " s"
              << std::endl;

    using State = Eigen::Matrix<double, NUM_STATES, 1>;
    std::vector<State, Eigen::aligned_allocator<State>> history;
    history.reserve(NUM_TICKS);

    Eigen::Matrix<double, NUM_MEASUREMENTS, 1> measure;

    measure.setZero();
    for (int k = 0; k < NUM_TICKS; ++k)
    {
        measure(0,0) = baro_height[k];
        ekf.set_measurement(measure);
        ekf.tick();
        history.push_back(ekf.get_x());
    }

    const Eigen::IOFormat csv(Eigen::FullPrecision, Eigen::DontAlignCols, ",", ",");
    std::ofstream out("x_est.csv");
    out << "x,vx,y,vy,z,vz\n";
    for (const State &s : history)
    {
        out << s.transpose().format(csv) << "\n";
    }

    
    return 0;
}



