#ifndef GMPHD_H
#define GMPHD_H

#include <vector>
#include <Eigen/Dense>

struct GaussianComponent {
    double weight;
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
};

struct Measurement {
    double time;
    Eigen::Vector2d value; // [theta, phi]
};

class GMPHD {
public:
    GMPHD(double p_survival, double p_detection, double clutter_intensity,
          const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q,
          const Eigen::MatrixXd& R, double initial_time);

    void processMeasurement(const Eigen::Vector2d& measurement, double time);
    std::vector<Eigen::VectorXd> extractStates(double threshold);
    std::vector<std::vector<int>> associateLinesOfSight(const std::vector<Measurement>& measurements, double distance_threshold);

private:
    double p_survival_;
    double p_detection_;
    double clutter_intensity_;
    Eigen::MatrixXd F_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
    double current_time_;
    std::vector<GaussianComponent> components_;

    void predict(double time);
    void update(const Eigen::Vector2d& measurement);
    void addBirthComponents(const Eigen::Vector2d& measurement);
    void prune(double threshold);
    Eigen::Vector2d h(const Eigen::VectorXd& state);
    Eigen::MatrixXd jacobian(const Eigen::VectorXd& state);
    double gaussianDensity(const Eigen::Vector2d& x, const Eigen::Vector2d& mean, const Eigen::MatrixXd& cov);
};

#endif