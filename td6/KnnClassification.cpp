
#include "KnnClassification.hpp"
#include <iostream>
#include <ANN/ANN.h>


KnnClassification::KnnClassification(int k, Dataset *dataset, int col_class)
: Classification(dataset, col_class) {
    // TODO Exercise 1

    m_k = k;
    int maxPts = dataset->getNbrSamples();
    int m_dim = dataset->getDim() - 1;
    m_dataPts = annAllocPts(maxPts, m_dim);
    m_col_class = col_class;

    int nPts = 0;

    while (nPts < maxPts){
        double* new_point = new double[m_dim];
        for(int i = 0; i < m_dim + 1; i++) {
            if(i == col_class) continue;
            new_point[i - (i>col_class)] = dataset->getInstance(nPts)[i];
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

KnnClassification::~KnnClassification() {
    // TODO Exercise 1
    delete[] m_dataPts;
    delete m_kdTree;
}

int KnnClassification::Estimate(const ANNpoint &x, double threshold) const {
    // TODO Exercise 2
    int nnIdx[m_k];
    double distance[m_k];
    m_kdTree->annkSearch(x, m_k, nnIdx, distance, 0.0001);

    double count = 0;

    for(int i = 0; i < m_k; i++)
        count += m_dataset->getInstance(nnIdx[i])[m_col_class];

    return count > (double) m_k * threshold;
}

int KnnClassification::getK() const {
    return m_k;
}

ANNkd_tree *KnnClassification::getKdTree() {
    return m_kdTree;
}
