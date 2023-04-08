
#include "KnnClassification.hpp"
#include <iostream>
#include <ANN/ANN.h>


KnnClassification::KnnClassification(int k, Dataset *dataset, int col_class)
: Classification(dataset, col_class) {
    // TODO Exercise 1
}

KnnClassification::~KnnClassification() {
    // TODO Exercise 1
}

int KnnClassification::Estimate(const ANNpoint &x, double threshold) const {
    // TODO Exercise 2
}

int KnnClassification::getK() const {
    return m_k;
}

ANNkd_tree *KnnClassification::getKdTree() {
    return m_kdTree;
}
