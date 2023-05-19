# 📚 INF442 - Algorithms for Data Analysis in C++ (École Polytechnique)

## 📊 About the course

Repository dedicated to my solutions to the practical sessions of the INF442 - Algorithms for Data Analysis in C++ course of École Polytechnique from professor Steve Oudot (steve.oudot@polytechnique.edu). 

The goal pursued by this course is twofold: first, to get acquainted with some of the standard techniques in data analysis and machine learning; second, to acquire some expertise in C/C++ programming, so that students can then adapt existing low-level implementations to their needs.

## 📝 Content description

Each class is indicated by "td" and its number. Here is a short description of each of them.

- TD 1 (Introduction to Unix and basic statistics in C++) - Implementation of simple statistics functions to get acquainted with C++.

- TD 2 (kNN and K-Dimensional Tree (kd-tree)) - Study of the famous Iris data set using kNN technique. Implementation in C++ for finding the nearest neighbor of a query point in a database using various searching strategies. 

- TD 3 (k-Means) - Implementation in C++ of the k-means algorithm using Voronoi partition, trying two different initializations such as random-partition, Forgy and k-means++ initializations. 

- TD 4 (Agglomerative hierarchical clustering) - Study of several datasets in python using scikit-learn to perform hierarchical clustering (dendrogram and linkage). Implementation of a single-linkage hierarchical clustering algorithm using union-find data structure in C++.

- TD 5 (Density Estimation) - Study of MNIST and other datasets using kernel density estimators using scikit-learn (gaussian and tophat). Implementation of three different kernel density estimators (flat, Gaussian, and k-nn), also removing noise from the dataset by implementing the mean shift algorithm on the k-nearest neighbor kernel in C++.

- TD 6 (kNN for classification) - Implementation of the supervised learning algorithm k-nearest neighbors (k-NN) in C++ as well as some classific classification performance metrics (such as confusion matrix).

- TD 7 (kNN for regression and Linear Regression) - Implementation of both k-NN regressor and linear regressor algorithms in C++, as well as performance comparison.

- TD 8 (Support Vector Machines) - Implementation of Soft Margin Kernel SVMs by solving the dual problem via a projected gradient ascent in C++.

- TD 9 (Regression with a One-Layer Perceptron) - Implementation of a 1-layer perceptron with one hidden layer in C++.