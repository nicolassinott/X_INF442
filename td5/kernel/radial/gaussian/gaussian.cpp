#include <cmath> // for pow, atan, should you need them
#include <iostream> // for cerr

#include <point.hpp>
#include <cloud.hpp>
#include <gaussian.hpp>

// TODO 2.1.2: implement volume, profile and guess_bandwidth
// HINTS: pi = std::atan(1) * 4.0, e^x is std::exp(x)
double gaussian::volume() const {
	int dimension = data->get_point(0).get_dim();
	return std::pow(2*M_PI, (double) dimension/2);
}

double gaussian::profile(double t) const {
	return std::exp(-t/2);
}

void gaussian::guess_bandwidth() {
	double estimated_std = data->standard_deviation();
	int n = data->get_n();
	bandwidth = 1.06 * (estimated_std / std::pow(n, (double) 1/5));
}
