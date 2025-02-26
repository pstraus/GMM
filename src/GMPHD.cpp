#include "GMPHD.h"
#include <cmath>
#include <algorithm>

GMPHD::GMPHD(double p_survival, double p_detection, double clutter_intensity,
             const Eigen::MatrixXd& F, const Eigen::MatrixXd& Q,
             const Eigen::MatrixXd& R, double initial_time)
    : p_survival_(p_survival), p_detection_(p_detection), clutter_intensity_(clutter_intensity),
      F_(F), Q_(Q), R_(R), current_time_(initial_time) {}

void GMPHD::processMeasurement(const Eigen::Vector2d& measurement, double time) {
    predict(time);
    addBirthComponents(measurement);
    update(measurement);
    prune(1e-4);
    current_time_ = time;
}

void GMPHD::predict(double time) {
    double dt = time - current_time_;
    if (dt <= 0) return;

    // For simplicity, assume F_ and Q_ are for dt=1, adjust in real scenarios
    for (auto& comp : components_) {
        comp.mean = F_ * comp.mean;
        comp.covariance = F_ * comp.covariance * F_.transpose() + Q_;
        comp.weight *= p_survival_;
    }
}

void GMPHD::addBirthComponents(const Eigen::Vector2d& measurement) {
    double theta = measurement(0);
    double phi = measurement(1);
    Eigen::Vector3d d(std::cos(theta) * std::cos(phi), std::sin(theta) * std::cos(phi), std::sin(phi));
    double r_values[] = {10.0, 20.0, 30.0};

    for (double r : r_values) {
        Eigen::VectorXd mean(6);
        mean.head<3>() = r * d;
        mean.tail<3>().setZero();
        Eigen::MatrixXd cov = Eigen::MatrixXd::Identity(6, 6);
        cov.block<3,3>(0,0) *= 100.0; // Large position uncertainty
        cov.block<3,3>(3,3) *= 1.0;   // Small velocity uncertainty
        components_.push_back({0.1, mean, cov});
    }
}

void GMPHD::update(const Eigen::Vector2d& measurement) {
    std::vector<GaussianComponent> new_components;
    double sum_weights = 0.0;
    std::vector<double> likelihoods;
    std::vector<GaussianComponent> detected;

    for (const auto& comp : components_) {
        Eigen::Vector2d z_pred = h(comp.mean);
        Eigen::MatrixXd H = jacobian(comp.mean);
        Eigen::MatrixXd S = H * comp.covariance * H.transpose() + R_;
        Eigen::MatrixXd K = comp.covariance * H.transpose() * S.inverse();
        Eigen::Vector2d innov = measurement - z_pred;

        GaussianComponent det_comp;
        det_comp.mean = comp.mean + K * innov;
        det_comp.covariance = comp.covariance - K * H * comp.covariance;
        double l = gaussianDensity(measurement, z_pred, S);
        det_comp.weight = p_detection_ * comp.weight * l;
        sum_weights += det_comp.weight;
        detected.push_back(det_comp);
        likelihoods.push_back(l);

        GaussianComponent miss_comp = comp;
        miss_comp.weight *= (1 - p_detection_);
        new_components.push_back(miss_comp);
    }

    double denom = clutter_intensity_ + sum_weights;
    for (auto& comp : detected) {
        comp.weight /= denom;
        new_components.push_back(comp);
    }

    components_ = new_components;
}

void GMPHD::prune(double threshold) {
    std::vector<GaussianComponent> pruned;
    for (const auto& comp : components_) {
        if (comp.weight > threshold) {
            pruned.push_back(comp);
        }
    }
    components_ = pruned;
}

std::vector<Eigen::VectorXd> GMPHD::extractStates(double threshold) {
    std::vector<Eigen::VectorXd> states;
    for (const auto& comp : components_) {
        if (comp.weight > threshold) {
            states.push_back(comp.mean);
        }
    }
    return states;
}

std::vector<std::vector<int>> GMPHD::associateLinesOfSight(const std::vector<Measurement>& measurements, double distance_threshold) {
    auto states = extractStates(0.5);
    std::vector<std::vector<int>> associations(states.size());

    for (size_t i = 0; i < states.size(); ++i) {
        Eigen::Vector3d p_t = states[i].head<3>();
        Eigen::Vector3d v = states[i].tail<3>();

        for (size_t j = 0; j < measurements.size(); ++j) {
            double dt = current_time_ - measurements[j].time;
            Eigen::Vector3d p_tm = p_t - v * dt;
            double theta = measurements[j].value(0);
            double phi = measurements[j].value(1);
            Eigen::Vector3d d(std::cos(theta) * std::cos(phi), std::sin(theta) * std::cos(phi), std::sin(phi));

            // Distance from point to line: |p_tm x d| since line passes through origin
            double dist = (p_tm.cross(d)).norm();
            if (dist < distance_threshold) {
                associations[i].push_back(j);
            }
        }
    }
    return associations;
}

Eigen::Vector2d GMPHD::h(const Eigen::VectorXd& state) {
    double x = state(0), y = state(1), z = state(2);
    double r = std::sqrt(x*x + y*y + z*z);
    if (r < 1e-6) return Eigen::Vector2d::Zero();
    return Eigen::Vector2d(std::atan2(y, x), std::asin(z / r));
}

Eigen::MatrixXd GMPHD::jacobian(const Eigen::VectorXd& state) {
    double x = state(0), y = state(1), z = state(2);
    double r2 = x*x + y*y + z*z;
    double r = std::sqrt(r2);
    double xy2 = x*x + y*y;
    double xy = std::sqrt(xy2);
    if (r < 1e-6 || xy < 1e-6) return Eigen::MatrixXd::Zero(2, 6);

    Eigen::MatrixXd H(2, 6);
    H.setZero();
    H(0, 0) = -y / xy2;
    H(0, 1) = x / xy2;
    H(1, 0) = -z * x / (r2 * xy);
    H(1, 1) = -z * y / (r2 * xy);
    H(1, 2) = xy / r2;
    return H;
}

double GMPHD::gaussianDensity(const Eigen::Vector2d& x, const Eigen::Vector2d& mean, const Eigen::MatrixXd& cov) {
    Eigen::Vector2d diff = x - mean;
    double exp_term = -0.5 * diff.transpose() * cov.inverse() * diff;
    double norm = std::pow(2 * M_PI, -1) * std::pow(cov.determinant(), -0.5);
    return norm * std::exp(exp_term);
}