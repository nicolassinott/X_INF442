// Include the library that defines input/output streams and operations on them
#include <iostream>
// Include the library with basic mathematical functions, e.g. sqrt
#include <cmath>
// Need that for memset
#include <cstring>

/********************
 * Helper functions *
 ********************/

// Read the data matrix from the standard input
void ReadMatrix (double** matrix, int rows, int columns) {
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new double[columns];
        for (int j = 0; j < columns; ++j)
            std::cin >> matrix[i][j];
    }
}

// Print an array on the standard output
void PrintArray (double values[], int length) {
    for (int i = 0; i < length - 1; i++)
        std::cout << values[i] << " ";

    std::cout << values[length - 1];
}

// Prepare an empty matrix
double** PrepareMatrix(int rows, int columns) {
    double** matrix = new double* [rows];
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new double[columns];
        for (int j = 0; j < columns; ++j)
            matrix[i][j] = 0;
    }

    return matrix;
}

/*****************************************************
 * Exercise 1: Mean, variance and standard deviation *
 *****************************************************/

/** 
 * This function computes the mean of the given array of values 
 * 
 * @param values the array with the values
 * @param length the length of the array 'values'
 * @return the mean of the values in the array
 */
double ComputeMean (double values[], int length) {
    double sum = 0;

    for(int i = 0; i < length; i++) sum += values[i];

    return sum / length;
}

/** 
 * This function computes the variance of the given array of values 
 * 
 * @param values the array with the values
 * @param length the length of the array 'values'
 * @return the variance of the values in the array
 */
double ComputeVariance (double values[], int length) {
    double mean = ComputeMean(values, length);
    // TODO

    double var = 0;

    for(int i = 0; i < length; i++) var += (values[i] - mean) * (values[i] - mean);

    return var / length;
}


/** 
 * This function computes the unbiased sample variance of the given
 * array of values 
 * 
 * @param values the array with the values
 * @param length the length of the array values
 * @return the variance of the values in the array
 */
double ComputeSampleVariance (double values[], int length) {
    // TODO
    // Note that you can call the functions defined above

    double sampleVariance = ComputeVariance(values, length);
    sampleVariance *= length;
    sampleVariance /= (length - 1);

    // Do not forget to replace this return by a correct one!
    return sampleVariance;
}

/** 
 * This function computes the standard deviation of the given
 * array of values 
 * 
 * @param values the array with the values
 * @param length the length of the array values
 * @return the variance of the values in the array
 */
double ComputeStandardDeviation (double values[], int length) {
    // TODO
    // Note that you can call the functions defined above

    // Do not forget to replace this return by a correct one!
    return std::sqrt(ComputeVariance(values, length));
}

/** 
 * This function computes the unbiased sample standard deviation
 * of the given array of values 
 * 
 * @param values the array with the values
 * @param length the length of the array values
 * @return the variance of the values in the array
 */
double ComputeSampleStandardDeviation (double values[], int length) {
    // TODO
    // Note that you can call the functions defined above

    // Do not forget to replace this return by a correct one!
    return std::sqrt(ComputeSampleVariance(values, length));
}

/*************************************
 * Exercise 2: Working with matrices *
 *************************************/

/**
 * This function prints a rectangular matrix on the standard output, 
 * placing each row on a separate line.  
 * 
 * @param matrix the matrix to print
 * @param rows the number of rows in the matrix
 * @param columns the number of columns
 */
void PrintMatrix (double** matrix, int rows, int columns) {

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            std::cout << matrix[i][j] << ' ';
        }
        std::cout << std::endl;
    }

    // TODO
}

/** 
 * This function extracts one row from a data matrix
 * 
 * @param matrix the matrix with the data
 * @param columns the number of columns in the matrix
 * @param index the index of the row to extract
 * @param row the array where the extracted values are to be placed
 */
void GetRow (double** matrix, int columns, int index, double row[]) {
    // Clean out completely the row 
    std::memset(row, 0, columns * sizeof(double));

    // TODO

    for(int i = 0; i < columns; i++){
        row[i] = matrix[index][i];
    }
}

/** 
 * This function extracts one column from a data matrix
 * 
 * @param matrix the matrix with the data
 * @param rows the number of rows in the matrix
 * @param index the index of the column to extract
 * @param column the array where the extracted values are to be placed
 */
void GetColumn (double** matrix, int rows, int index, double column[]) {
    // Clean out completely the column 
    std::memset(column, 0, rows * sizeof(double));

    // TODO
    for(int i = 0; i < rows; i++){
        column[i] = matrix[i][index];
    }
}

/****************************************************
 * Exercise 3: Covariance and correlation of arrays *
 ****************************************************/

/**
 * This function computes the covariance of two vectors of data of the same length
 * @param values1 the first vector
 * @param values2 the second vector
 * @param length the length of the two vectors
 * @return the covariance of the two vectors
 */
double ComputeCovariance(double values1[], double values2[], int length) {
    // TODO
    double coVar = 0;
    double mean1 = ComputeMean(values1,length);
    double mean2 = ComputeMean(values2,length);

    for(int i = 0; i < length; i++)
        coVar += (values1[i] - mean1) * (values2[i] - mean2);
    
    coVar /= length;


    // Do not forget to replace this return by a correct one!
    return coVar;
}

/**
 * This function computes the correlation of two vectors of data of the same length
 * 
 * @param values1 the first vector
 * @param values2 the second vector
 * @param length the length of the two vectors
 * @return the correlation of the two vectors
 */
double ComputeCorrelation(double values1[], double values2[], int length) {
    // TODO: Use computeCovariance() and functions already implemented
    // Do not forget to replace this return by a correct one!

    double coVar = ComputeCovariance(values1, values2, length);
    double std1 = ComputeStandardDeviation(values1, length);
    double std2 = ComputeStandardDeviation(values2, length);

    return coVar / (std2 * std1);
}

/***************************************************
 * Exercise 4: Covariance and correlation matrices *
 ***************************************************/

/**
 * This function computes the covariance matrix of the matrix provided as argument
 * 
 * @param data_matrix the input matrix 
 * @param rows the number of rows in the matrix
 * @param columns the number of columns in the matrix
 * @return the covariance matrix
 */
double** ComputeCovarianceMatrix (double** data_matrix, int rows, int columns) {
    // Initialise a square matrix
    double** matrix = PrepareMatrix(columns, columns);
    // Prepare temporary storage for columns
    double column1[rows], column2[rows];

    // TODO

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            GetColumn(data_matrix, columns, i, column1);
            GetColumn(data_matrix, columns, j, column2);

            matrix[i][j] = ComputeCovariance(column1, column2, columns);
        }
    }

    return matrix;
}

/**
 * This function computes the correlation matrix of the matrix provided as argument
 * 
 * @param data_matrix the input matrix 
 * @param rows the number of rows in the matrix
 * @param columns the number of columns in the matrix
 * @return the correlation matrix
 */
double** ComputeCorrelationMatrix (double** data_matrix, int rows, int columns) {
    // Initialise a square matrix
    double** matrix = PrepareMatrix(columns, columns);
    // Prepare temporary storage for columns
    double column1[rows], column2[rows];

    // TODO

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            GetColumn(data_matrix, columns, i, column1);
            GetColumn(data_matrix, columns, j, column2);

            matrix[i][j] = ComputeCorrelation(column1, column2, columns);
        }
    }

    return matrix;
}
