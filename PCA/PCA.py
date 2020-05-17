# -*- coding: utf-8 -*-
"""
Created on Sat May  16 18:46:08 2020

@author: Norbert Toth
"""


from sklearn import model_selection as ms; 
from sklearn import decomposition as decomp;   
import numpy as np;  
from matplotlib import pyplot as plt;  
import urllib;
from numpy import linalg as LA;  # importing the linear algebra library from numpy

numberOfComponents = 2;

url = 'https://raw.githubusercontent.com/notusedusername/machine_learning/master/winequality-red.csv';
label_row_number = 1;
attribute_cols = 10;
output_variable_col = 11;



raw_data = urllib.request.urlopen(url);
data = np.loadtxt(raw_data, delimiter=";", dtype = float, skiprows=label_row_number)
X = data[:,0:attribute_cols];  
y = data[:,output_variable_col];

p = X.shape[1];


X_train, X_test, y_train, y_test = ms.train_test_split(X, 
             y, test_size=0.2, random_state=2020);
#test-size: 20% test

pca = decomp.PCA();
pca.fit(X);


fig = plt.figure(1);
plt.title('Variance diagram');
var_ratio = pca.explained_variance_ratio_;
x_pos = np.arange(len(var_ratio))+1;
plt.xlabel('Main components');
plt.ylabel('Variance');
plt.bar(x_pos,var_ratio, align='center', alpha=0.5);
plt.show(); 


pca = decomp.PCA(n_components = 2)
pca.fit(X);
PC_train = pca.transform(X);
fig = plt.figure(2);
plt.title('DataPoints');
plt.scatter(PC_train[:,0],PC_train[:,1],c=y, cmap = 'magma');
plt.show();

pca = decomp.PCA(n_components=numberOfComponents);
pca.fit(X_train);
PC_test = pca.transform(X_test);
fig = plt.figure(3);
plt.title('Test dataset');
plt.scatter(PC_test[:,0],PC_test[:,1],c=y_test,cmap = 'ocean');
plt.show();



# Comparing the complete data and partial data PCA on test dataset
X_full = np.concatenate((X_train,X_test), axis = 0);
pca.fit(X_full);
test_pc_full = pca.transform(X_test);

fig = plt.figure(4);
plt.title('Comparing  Wine data by');
plt.xlabel('PC1');
plt.ylabel('PC2');
plt.scatter(PC_test[:,0],PC_test[:,1],s=50,c=y_test,
            cmap='cool',label='Test data PCA');
plt.scatter(test_pc_full[:,0],test_pc_full[:,1],s=50,c=y_test,
            cmap='inferno',marker='P',label='Complete data PCA');
plt.legend();            
plt.show();    




Cov = np.cov(np.transpose(X));  # computing the covariance matrix for attributes
Cov_eig, Cov_vec = LA.eigh(Cov);     # solving the eigenvalue-eigenvector problem
PC_wine = np.dot(X,Cov_vec);    # computing the principal components
Cov_pc = np.cov(np.transpose(PC_wine));  # checking the covariance matrix of PCs
dim = Cov.shape[0];  # dimension of the covariance matrix
tot_var = np.trace(Cov);  # total variance of the covariance matrix
diff = tot_var - np.sum(Cov_eig);  # checking the sum of the covariance matrix
Cov_eig = np.flipud(Cov_eig);  # eigenvalues in decreasing order
var_ratio = Cov_eig/tot_var;    # normalized variances
cumvar_ratio = np.cumsum(Cov_eig)/tot_var;    # normalized cumulative variances
fig = plt.figure(6); # Explained cumulative variance ratio plot
plt.title('Explained cumulative variance ratio plot');
x_pos = np.arange(dim);
#plt.xticks(x_pos,x_pos+1);
plt.xlabel('Principal Components');
plt.ylabel('Cumulative variance');
plt.plot(x_pos,cumvar_ratio);
plt.show(); 

# 1 komponens használatával 95% közeli
# 2 komponens több mint 99%
# 8-10 komponens: a különbség kerekítési hiba mértékű