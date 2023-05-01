#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>

#include "kernel.hpp"
#include "confusion_matrix.hpp"
#include "svm.hpp"

SVM::SVM(Dataset* dataset, int col_class, Kernel K):
    col_class(col_class), kernel(K) {
    train_labels = std::vector<int>(dataset->GetNbrSamples());
    train_features = std::vector<std::vector<double>>(dataset->GetNbrSamples(), std::vector<double>(dataset->GetDim() - 1));
    // Exercise 2: put the correct columns of dataset in train_labels and _features AGAIN!
    // BEWARE: transform 0/1 labels to -1/1

    for(int i = 0; i < dataset->GetNbrSamples(); i++){
        std::vector<double> row = dataset->GetInstance(i);
        for(int j = 0; j < dataset->GetDim()-1; j++){
            if(j == col_class) train_labels[i] = 2*row[j] - 1;
            train_features[i][j] = row[j + (j >= col_class)];
        }
    }

    compute_kernel();
}

SVM::~SVM() {
}

void SVM::compute_kernel() {
    const int n = train_features.size();
    alpha = std::vector<double>(n);
    computed_kernel = std::vector<std::vector<double>>(n, std::vector<double>(n));

    // Exercise 2: put y_i y_j k(x_i, x_j) in the (i, j)th coordinate of computed_kernel

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            computed_kernel[i][j] = train_labels[i]*train_labels[j]*kernel.k(train_features[i], train_features[j]);
        }
    }
}

void SVM::compute_beta_0(double C) {
    // count keeps track of the number of support vectors (denoted by n_s)
    int count = 0;
    beta_0 = 0.0;
    // Exercise 3
    // Use clipping_epsilon < alpha < C - clipping_epsilon instead of 0 < alpha < C
    const int n = train_features.size();

    for(int s = 0; s < n; s++){ 
        if(clipping_epsilon < alpha[s] && alpha[s] < C-clipping_epsilon){
            count++;
            for(int i = 0; i < n; i++) beta_0 += alpha[i]*train_labels[i]*kernel.k(train_features[i], train_features[s]);// - train_labels[s]; // computed_kernel[i][s]
            beta_0 -= train_labels[s];
        }
    }

    // This performs 1/n_s
    beta_0 /= count;
}

void SVM::train(const double C, const double lr) {
    // Exercise 4
    // Perform projected gradient ascent
    // While some alpha is not clipped AND its gradient is above stopping_criterion
    // (1) Set stop = false
    // (2) While not stop do
    // (2.1) Set stop = true
    // (2.2) For i = 1 to n do
    // (2.2.1) Compute the gradient - HINT: make good use of computed_kernel
    // (2.2.2) Move alpha in the direction of the gradient - eta corresponds to lr (for learning rate)
    // (2.2.3) Project alpha in the constraint box by clipping it
    // (2.2.4) Adjust stop if necessary
    // (3) Compute beta_0

    const int n = train_features.size();

    bool stop = 0;

    while(!stop){
        stop = 1;

        for(int i = 0; i < n; i++){
            double gradient = 1;
            for(int j = 0; j < n; j++)
                gradient -= alpha[j]*train_labels[i]*train_labels[j]*kernel.k(train_features[i], train_features[j]);//computed_kernel[i][j];
            alpha[i] += lr * gradient;
            
            bool clip_necessary = 0; double alpha_start = alpha[i];
            alpha[i] = std::max(std::min(alpha[i],C), 0.0);
            if(alpha_start != alpha[i]) clip_necessary = 1;

            if(!clip_necessary && std::abs(gradient) > stopping_criterion) stop = 0;
        }   
    }
    // Update beta_0
    compute_beta_0(C);
}

int SVM::f_hat(const std::vector<double> x) {
    // Exercise 5
    double pred = 0;
    int n = train_features.size();

    for(int i = 0; i < n; i++){
        pred += alpha[i]*train_labels[i]*kernel.k(train_features[i],x);
    }

    pred -= beta_0;

    return 2*(pred > 0) - 1;
}

ConfusionMatrix SVM::test(const Dataset* test) {
    // Collect test_features and test_labels and compute confusion matrix
    std::vector<double> test_labels (test->GetNbrSamples());
    std::vector<std::vector<double>> test_features (test->GetNbrSamples(), std::vector<double>(test->GetDim() - 1));
    ConfusionMatrix cm;

    // Exercise 6
    // Put test dataset in features and labels
    // Use f_hat to predict and put into the confusion matrix
    // BEWARE: transform -1/1 prediction to 0/1 label

    for(int i = 0; i < test->GetNbrSamples(); i++){
        std::vector<double> row = test->GetInstance(i);
        for(int j = 0; j < test->GetDim()-1; j++){
            if(j == col_class) test_labels[i] = 2*row[j] - 1;
            test_features[i][j] = row[j + (j >= col_class)];
        }
    }

    for(int i = 0; i < test->GetNbrSamples(); i++){
        int true_label = (test_labels[i] + 1)/2;
        int predicted_label = (f_hat(test_features[i]) + 1)/2;
        cm.AddPrediction(true_label, predicted_label);
    }

    return cm;
}

int SVM::get_col_class() const {
    return col_class;
}

Kernel SVM::get_kernel() const {
    return kernel;
}

std::vector<int> SVM::get_train_labels() const {
    return train_labels;
}

std::vector<std::vector<double>> SVM::get_train_features() const {
    return train_features;
}

std::vector<std::vector<double>> SVM::get_computed_kernel() const {
    return computed_kernel;
}

std::vector<double> SVM::get_alphas() const {
    return alpha;
}

double SVM::get_beta_0() const {
    return beta_0;
}

void SVM::set_alphas(std::vector<double> alpha) {
    this->alpha = alpha;
}
