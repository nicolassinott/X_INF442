#include <cmath> // for pow, atan, should you need them

#include <point.hpp>
#include <flat.hpp>

// TODO 2.1.1: implement volume and profile
// HINT: pi = std::atan(1) * 4.0
double flat::volume() const {
	int dimension = data->get_point(0).get_dim();
	return pow(M_PI, (double) dimension/2) / std::tgamma((double)dimension/2 + 1);
}

double flat::profile(double t) const {
	return t <= 1;
}
