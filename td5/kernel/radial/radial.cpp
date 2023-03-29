#include <cmath> // for pow, should you need it

#include <point.hpp>
#include <cloud.hpp>
#include <radial.hpp>

// TODO 2.1: density and radial constructor
// Use profile and volume... although it will only be implemented in the "sisters" classes
// Use kernel's constructor

// radial::radial(cloud& _data, double _bandwidth){

// }

radial::radial ( cloud* _data, double _bandwidth ): kernel::kernel(_data) {
	bandwidth = _bandwidth;
} 


double radial::density(const point& p) const
{
	return 0.0;
}
