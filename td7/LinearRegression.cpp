#include "LinearRegression.hpp"
#include "Dataset.hpp"
#include "Regression.hpp"
#include<iostream>
#include<cassert>


LinearRegression::LinearRegression( Dataset* dataset, int col_regr ) 
: Regression(dataset, col_regr) {
	m_beta = NULL;
	SetCoefficients();
}

LinearRegression::~LinearRegression() {
	if (m_beta != NULL) {
		m_beta->resize(0);
		delete m_beta;
	}
}


Eigen::MatrixXd LinearRegression::ConstructMatrix() {
	// TODO Exercise 1
	int n_samples = m_dataset->GetNbrSamples();
	int n_dim = m_dataset->GetDim();

	Eigen::MatrixXd X(n_samples, n_dim);
	for(int i = 0; i < n_samples; i++) X(i,0) = (double)1;

	for(int i = 0; i < n_samples; i++){
		std::vector<double> x_vec = m_dataset->GetInstance(i);
		for(int j = 0; j < n_dim-1; j++) X(i,j+1) = x_vec[j + (j>=m_col_regr)];
	}

	return X;
}

Eigen::VectorXd LinearRegression::ConstructY() {
	// TODO Exercise 1

	int n_samples = m_dataset->GetNbrSamples();
	int n_dim = m_dataset->GetDim();

	Eigen::VectorXd y(n_samples);

	for(int i = 0; i < n_samples; i++){
		std::vector<double> sample = m_dataset->GetInstance(i);
		y[i] = sample[m_col_regr];
	}

	// replace this command with what you compute as a vector y.
	return y;
}

void LinearRegression::SetCoefficients() {
	// TODO Exercise 2
	Eigen::MatrixXd X = ConstructMatrix();
	Eigen::VectorXd y = ConstructY();
	Eigen::MatrixXd Xt = X.transpose();


	Eigen::MatrixXd XtX = Xt * X;
	// Eigen::MatrixXd Id(XtX.cols(), XtX.cols());
	// for(int i = 0; i < XtX.cols(); i++) Id(i,i) = (double) 1;
	// Eigen::MatrixXd XtXINV = XtX.fullPivHouseholderQr().solve(Id); // (XtX.cols(), XtX.cols());
	Eigen::MatrixXd XtXINV = XtX.inverse();

	// Eigen::VectorXd b_temp = X.transpose() * y;
	Eigen::VectorXd b = XtXINV * X.transpose() * y;

	m_beta = new Eigen::VectorXd(b.rows());
	*m_beta = b;
}

const Eigen::VectorXd* LinearRegression::GetCoefficients() const {
	if (!m_beta) {
		std::cout<<"Coefficients have not been allocated."<<std::endl;
		return NULL;
	}
	return m_beta;
}

void LinearRegression::ShowCoefficients() const {
	if (!m_beta) {
		std::cout<<"Coefficients have not been allocated."<<std::endl;
		return;
	}
	
	if (m_beta->size() != m_dataset->GetDim()) {  // ( beta_0 beta_1 ... beta_{d} )
		std::cout<< "Warning, unexpected size of coefficients vector: " << m_beta->size() << std::endl;
	}
	
	std::cout<< "beta = (";
	for (int i=0; i<m_beta->size(); i++) {
		std::cout<< " " << (*m_beta)[i];
	}
	std::cout << " )"<<std::endl;
}

void LinearRegression::PrintRawCoefficients() const {
	std::cout<< "{ ";
	for (int i=0; i<m_beta->size()-1; i++) {
		std::cout<< (*m_beta)[i]  << ", ";
	}
	std::cout << (*m_beta)[m_beta->size()-1];
	std::cout << " }"<<std::endl;
}

void LinearRegression::SumsOfSquares( Dataset* dataset, double& ess, double& rss, double& tss ) const {
	assert(dataset->GetDim()==m_dataset->GetDim());
	// TODO Exercise 4
	int n_samples = dataset->GetNbrSamples();
	int n_dim = dataset->GetDim();

	// Getting X
	Eigen::MatrixXd X(n_samples, n_dim);
	for(int i = 0; i < n_samples; i++) X(i,0) = (double)1;

	for(int i = 0; i < n_samples; i++){
		std::vector<double> x_vec = dataset->GetInstance(i);
		for(int j = 0; j < n_dim-1; j++) X(i,j+1) = x_vec[j + (j>=m_col_regr)];
	}

	// Getting y
	Eigen::VectorXd y(n_samples);
	for(int i = 0; i < n_samples; i++){
		std::vector<double> sample = dataset->GetInstance(i);
		y[i] = sample[m_col_regr];
	}

	Eigen::VectorXd y_pred(n_samples);
	for(int i = 0; i < n_samples; i++){
		Eigen::VectorXd x = X.row(i);
		y_pred(i) = m_beta->dot(x);
	}

	rss = 0;
	for(int i = 0; i < n_samples; i++) rss += (y_pred(i) - y(i)) * (y_pred(i) - y(i)); 

	double y_bar = 0;
	for(int i = 0; i < n_samples; i++) y_bar += y(i);
	y_bar /= n_samples;

	tss = 0;
	for(int i = 0; i < n_samples; i++) tss += (y(i) - y_bar) * (y(i) - y_bar); 

	ess = 0;
	for(int i = 0; i < n_samples; i++) ess += (y_pred(i) - y_bar) * (y_pred(i) - y_bar); 
}

double LinearRegression::Estimate( const Eigen::VectorXd & x ) const {
	// TODO Exercise 3
	int n_dim = m_dataset->GetDim();
	Eigen::VectorXd beta = *m_beta;

	double result = beta(0);
	for(int i = 0; i <n_dim-1; i++) result+= beta(i+1) * x(i);

	return result;
}
