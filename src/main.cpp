#include "GMPHD.h"
#include <iostream>

int main() {
    // System parameters
    Eigen::MatrixXd F(6, 6);
    F << 1,0,0,1,0,0,
         0,1,0,0,1,0,
         0,0,1,0,0,1,
         0,0,0,1,0,0,
         0,0,0,0,1,0,
         0,0,0,0,0,1;

    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(6, 6) * 0.1;
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(2, 2) * 0.01;

    GMPHD filter(0.99, 0.95, 1.0, F, Q, R, 0.0);

    // Example measurements
    std::vector<Measurement> measurements = {
        {1.0, Eigen::Vector2d(0.1, 0.2)},
        {2.0, Eigen::Vector2d(0.15, 0.21)},
        {3.0, Eigen::Vector2d(0.14, 0.19)}
    };

    for (const auto& m : measurements) {
        filter.processMeasurement(m.value, m.time);
    }

    // Extract states
    auto states = filter.extractStates(0.5);
    std::cout << "Number of targets: " << states.size() << "\n";
    for (size_t i = 0; i < states.size(); ++i) {
        std::cout << "Target " << i << ": " << states[i].transpose() << "\n";
    }

    // Associate lines of sight
    auto associations = filter.associateLinesOfSight(measurements, 5.0);
    for (size_t i = 0; i < associations.size(); ++i) {
        std::cout << "Target " << i << " associated with measurements: ";
        for (int idx : associations[i]) {
            std::cout << idx << " ";
        }
        std::cout << "\n";
        if (associations[i].size() > 1) {
            std::cout << "Lines of sight merge at Target " << i << "\n";
        }
    }

    return 0;
}