# Machine learning implementation of basic algorithms in small projects

##Algorithms used:
*Linear regression
*Logistic regression
*SVM (using LIBSVM in MATLAB) and using Sklearn in python
*Feedforward neural network (Using MATLAB NN toolbox, using sklearn, also implemented from scratch)
*PCA (principal component analysis)
*FCA/LDA (fisher linear discriminant analysis)
*K-means clustering
*linkage clustering
*KNN
*batch gradient decent
*Stocastic gradient decent 
*Effect of regularization 
*Naive bayes

# Folder description

## Classfication

Poker data is used for classification (https://archive.ics.uci.edu/ml/datasets/Poker+Hand)
*Each poker hand is ranked into 10 classes based on its importance. Total data: >1 million instances
*Input: 5 cards rank and suits (total 10 features)
*staright forward SVM implemetation didn't work. So new features were generate. New features were the difference
between suits and ranks. So total new features were 5 choose 2 + 5 choose 2. 
*SVM implementation on new features gave good results. Test set accuracy was >90. (Need to tune gamma value for SVM
with RBF kernel, also used LIBSVM library) 
*Original features (5 cards rank and suits gave good results with Feed forward neural network. 
Achieved 99.89% accuracy on the test set


Wine data classification(https://archive.ics.uci.edu/ml/datasets/Wine)
* Used linear classification and Fisher discrimant analysis for the 3 class classifier. 


## Linear regression
* Noisy sine wave data is generate. 
* Linear regression with different order of polynomial and different regularization was fitted. 
* Coded in python and matlab using normal equation
* Effect of regularization, order of polynomial and availability of data points analyzed


## KNN 
*Battery state of charge is estimated using KNN. Feature vector is generated using some other technique (not 
giving here, work submited for publication)
* Features are used to estimate battery state of charge using matlab inbuild methods. 

## PCA Kmeans Linkage
* Healthy and unhealty battery data is used for feature extraction (not giving here, work submited for publication)
* PCA performed on the features, and Kmeans and linkage clustering methods are used to cluster out the unhealthy
batteries
*K-means didn't performed well here as it is more suitable for spherical clusters. 
* Linkage clusters worked well as clusters were quite far away from each other. 

## FNN from scratch 
*Feed forward neural network is implemented from scratch for better understanding of feedforward and backpropagation
algorithm. 
* Sigmoid and tahn+softmax activation functions used in different implementation. 
