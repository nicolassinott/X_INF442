
#include "KnnRegression.hpp"
#include<iostream>
#include <ANN/ANN.h>


KnnRegression::KnnRegression(int k, Dataset* dataset, int col_regr)
: Regression(dataset, col_regr) {
	m_k = k;
 	// TODO Exercise 5
	int maxPts = dataset->GetNbrSamples();
    int m_dim = dataset->GetDim() - 1;
    m_dataPts = annAllocPts(maxPts, m_dim);
    m_col_regr = col_regr;

    int nPts = 0;

    while (nPts < maxPts){
        double* new_point = new double[m_dim];
        for(int i = 0; i < m_dim + 1; i++) {
            if(i == col_regr) continue;
            new_point[i - (i>col_regr)] = dataset->GetInstance(nPts)[i];
        } 
        m_dataPts[nPts] = new_point;
        nPts++;
    } 

    m_kdTree = new ANNkd_tree(
        m_dataPts,
        maxPts, 
        m_dim
        ); 
}

KnnRegression::~KnnRegression() {
	// TODO Exercise 5
	delete[] m_dataPts;
    delete m_kdTree;
}

double KnnRegression::Estimate(const Eigen::VectorXd & x) const {
	assert(x.size()==m_dataset->GetDim()-1);
	// TODO Exercise 6
	int nnIdx[m_k];
    double distance[m_k];
	
	double* x_double = new double[x.rows()];
	for(int i = 0; i < x.rows(); i++) x_double[i] = x(i);

    m_kdTree->annkSearch(x_double, m_k, nnIdx, distance, 0.0001);

    double count = 0;

    for(int i = 0; i < m_k; i++)
        count += m_dataset->GetInstance(nnIdx[i])[m_col_regr];

    return count/= m_k;
}

int KnnRegression::GetK() const {
	return m_k;
}

ANNkd_tree* KnnRegression::GetKdTree() const {
	return m_kdTree;
}
