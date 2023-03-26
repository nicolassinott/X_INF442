#include <iostream>
#include <cassert>
#include <cmath>	// for sqrt, fabs
#include <cfloat>	// for DBL_MAX
#include <cstdlib>	// for rand, srand
#include <ctime>	// for rand seed
#include <fstream>
#include <cstdio>	// for EOF
#include <string>
#include <algorithm>	// for count
#include <vector>

using std::rand;
using std::srand;
using std::time;
using std::min;

class point
{
    public:

        static int d;
        double *coords;
        int label;

		point(){
			coords = new double[d];
			label = 0;
			for(int i = 0; i < d; i++) coords[i] = 0;  
		}
		 
		~point(){
			delete[] coords;
		}

		void print() const {
			for(int i = 0; i < d - -1; i++){
				std::cout << coords[i] << '\t';
			}
			std::cout << coords[d-1] << std::endl;
		}

		double squared_dist(const point &q) const {
			double dist = 0;
			
			for(int i = 0; i < d; i++){
				dist += (coords[i] - q.coords[i]) * (coords[i] - q.coords[i]); 
			}

			return dist;
		}

		double get_coordinate(int i){
			return coords[i];
		}

		void add_coordinates(point& q) {
			for(int i = 0; i < d; i++) coords[i] += q.get_coordinate(i);
		}

		void divide_coordinates(int n){
			for(int i = 0; i < d; i++) coords[i] /= n;
		}

		void copy_coordinates(point& q){
			for(int i = 0; i < d; i++) coords[i] = q.get_coordinate(i);
		}
};

int point::d;


class cloud
{
	private:

	int d;
	int n;
	int k;

	// maximum possible number of points
	int nmax;

	point *points;
	point *centers;


	public:

	cloud(int _d, int _nmax, int _k)
	{
		d = _d;
		point::d = _d;
		n = 0;
		k = _k;

		nmax = _nmax;

		points = new point[nmax];
		centers = new point[k];

		srand(time(0));
	}

	~cloud()
	{
		delete[] centers;
		delete[] points;
	}

	void add_point(const point &p, int label)
	{
		for(int m = 0; m < d; m++)
		{
			points[n].coords[m] = p.coords[m];
		}

		points[n].label = label;

		n++;
	}

	int get_d() const
	{
		return d;
	}

	int get_n() const
	{
		return n;
	}

	int get_k() const
	{
		return k;
	}

	point& get_point(int i)
	{
		return points[i];
	}

	point& get_center(int j)
	{
		return centers[j];
	}

	void set_center(const point &p, int j)
	{
		for(int m = 0; m < d; m++)
			centers[j].coords[m] = p.coords[m];
	}

	double intracluster_variance() const
	{
		double variance = 0;

		for(int i = 0; i < n; i++){
			int cluster = points[i].label;
			variance += points[i].squared_dist(centers[cluster]);
		}
		return variance / n;
	}

	int set_voronoi_labels()
	{
		int changed_labels = 0;

		for(int i = 0; i < n; i++){
			double dist = DBL_MAX;
			int label = -1;

			for(int j = 0; j < k; j++){
				
				double curr_dist = points[i].squared_dist(centers[j]);

				if(curr_dist >= dist) continue;
				
				dist = curr_dist;
				label = j;
			}

			if(points[i].label != label){
				points[i].label = label;
				changed_labels++;
			}
		}

		return changed_labels;
	}

	void set_centroid_centers()
	{
		std::vector<point> new_centers(k);
		std::vector<int> n_elements(k);

		for(int i = 0; i < k; i++){
			n_elements[i] = 0;
		}

		for(int i = 0; i < n; i++){
			int label = points[i].label;
			n_elements[label]++;
			new_centers[label].add_coordinates(points[i]);
		}

		for(int i = 0; i < k; i++){
			if(n_elements[i] == 0) continue;
			new_centers[i].divide_coordinates(n_elements[i]);
			centers[i].copy_coordinates(new_centers[i]);	
		}
	}

	void init_random_partition()
	{
		for(int i = 0; i < n; i++){
			points[i].label = rand() % k;
		}
		set_centroid_centers();
	}

	void lloyd()
	{
		int changes = 1;
		while(changes > 0){
			changes = set_voronoi_labels();
			set_centroid_centers();
		}
	}

	void init_forgy()
	{
		std::vector<int> called(n);

		for(int i = 0; i < k; i++){
			int rand_number = rand() % n;
			while(called[rand_number]){
				rand_number = rand() % n;
			}

			called[rand_number] = 1;
			centers[i].copy_coordinates(points[rand_number]);
		}
	}

	void init_plusplus()
	{
		std::vector<point> new_centers(k);
		int first_center = rand() % n;
		new_centers[0].copy_coordinates(points[first_center]);

		for(int i = 1; i < k; i++){
			// Updates the distance array
			std::vector<double> distances(n);
			double total_dist = 0;
			for(int j = 0; j < n; j++){
				distances[j] = DBL_MAX;
				for(int l = 0; l < i; l++){
					double new_dist = points[j].squared_dist(new_centers[l]);
					distances[j] = std::min(distances[j], new_dist);
				}
				total_dist += distances[j];
			}
			for(int j = 0; j < n; j++) distances[j] /= total_dist;

			
			// Perform a search to find index
			int index_center = -1;
			double random_number = (double) rand() / RAND_MAX;

			double curr_density = 0;

			for(int j = 0; j < n; j++){
				curr_density += distances[j];
				if (curr_density >= random_number){
					index_center = j;
					break;
				} 
			}

			// Update the new_centers list
			new_centers[i].copy_coordinates(points[index_center]);
		}

		for(int i = 0; i < k; i++){
			centers[i].copy_coordinates(new_centers[i]);
		}
	}
};
