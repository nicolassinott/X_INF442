#include <cmath> // for pow, should you need it

#include <point.hpp>
#include <cloud.hpp>
#include <radial.hpp>

// TODO 2.1: density and radial constructor
// Use profile and volume... although it will only be implemented in the "sisters" classes
// Use kernel's constructor

radial::radial ( cloud* _data, double _bandwidth ): kernel::kernel(_data) {
	bandwidth = _bandwidth;
} 

double radial::density(const point& p) const
{
	double c = 1 / volume();
	int n = data->get_n();
	int d = data->get_point(0).get_dim();

	double constant = (double) c / (n * std::pow(bandwidth, d));

	double profile_sum = 0;

	for(int i = 0; i < n; i++){
		double dist_squared = p.dist(data->get_point(i)); dist_squared = dist_squared * dist_squared;
		dist_squared /= pow(bandwidth, 2);
		profile_sum += profile(dist_squared);
	}


	return constant * profile_sum;
}