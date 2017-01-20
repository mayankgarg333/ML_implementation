# Ploting the input data and simple curve fit 


####
## Import libraries
####

import os
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

####
## load data
####
filename='data/data100.npz'
locals().update(np.load(filename))

####
## plot data
####
plt.figure(1, figsize=(10,10))
h1,=plt.plot(x,y, 'r-')
plt.axis([0, 15, -1.2, 1.2])
h2,=plt.plot(x,t, 'go')
plt.axis([0, 15, -1.2, 1.2])
plt.xlabel('x value')
plt.ylabel('y value')
t1=plt.title('Data Visulization')
plt.legend([h1,h2],['Correct plot','Noisy data'])
plt.setp([h1,h2], linewidth=3.0)
plt.rcParams.update({'font.size': 20})
plt.show()

####
## Linear regression curve fit with the function: f= w0+w1.x+w2.x^2 ... +wm-1*x^m-1
## USing the linear regression model given at page 142 Bisop
## X: defined as the matrix of all observation and all features (all input data)
## Each roq of X is a data point. So for any data point estimated y will be W' * X' (row)
## X will have the dimension of: N*m (number of row are data point, and features are column)
## W: weight vector of m entries. 
## W= inv(X' X ) * X' t ; t is actual data points with noise 
## the equation is calculated using error minimization
####

# create X:

M=5;                            # CHANGE THIS TO CHECK THE EFFECT OF DIFFERENT

X=np.array([])
for i in range(M):
    X=np.vstack([X,x**i]) if X.size else x**i    # use of vstack and hstack for matlab type manuplation 
X=X.T
W=np.dot(np.linalg.pinv(np.dot(X.T,X)),np.dot(X.T,t))
y_pred=np.dot(W.T,X.T)


###
## Ploting the actual data and predicted data
###

f2=plt.figure(2, figsize=(10,10))
h1,=plt.plot(x,y, 'r-')
plt.axis([0, 15, -1.2, 1.2])
h2,=plt.plot(x,t, 'go')
plt.axis([0, 15, -1.2, 1.2])
h3,=plt.plot(x,y_pred, 'b-')
plt.axis([0, 15, -1.2, 1.2])
plt.xlabel('x value')
plt.ylabel('y value')
t1=plt.title('Data Visulization')
plt.setp([h1,h2,h3], linewidth=3.0)
plt.legend([h1,h2,h3],['Correct plot','Noisy data','Predcition plot'])
plt.rcParams.update({'font.size': 20})
plt.show()



###
## Effect of M value (Effect of the order of the polynomial to fit)
###

# Y will hold the prediction for each M value
Y=np.array([])
for M in range(1,11):
    X=np.array([])
    for i in range(M):
        X=np.vstack([X,x**i]) if X.size else x**i    # use of vstack and hstack for matlab type manuplation 
    X=X.T
    I=np.linalg.pinv(np.dot(X.T,X)) if np.dot(X.T,X).shape else 1/np.dot(X.T,X)
    W=np.dot(I,np.dot(X.T,t))
    y_pred=np.dot(W.T,X.T)
    Y=np.vstack([Y,y_pred]) if Y.size else y_pred
    
    
f3=plt.figure(3, figsize=(20,20))
for M in range(1,11):
    p=plt.subplot2grid((2,5), ((M-1)/5,(M-1)%5))
    h1,=p.plot(x,y, 'r-')
    p.axis([0, 15, -1.2, 1.2])
    h2,=p.plot(x,t, 'go')
    p.axis([0, 15, -1.2, 1.2])
    y_pred=Y[0,:]
    Y=Y[1:,:]
    h3,=p.plot(x,y_pred, 'b-')
    p.axis([0, 15, -1.2, 1.2])
    plt.xticks(np.arange(-0, 15, 4))
    plt.setp([h1,h2,h3], linewidth=3.0)
    #p.axis([0, 15, -1.2, 1.2]).title('Data Visulization')
    #p.axis([0, 15, -1.2, 1.2]).legend([h1,h2,h3],['Correct plot','Noisy data','Predcition plot'])
    p.set_title("M : "+str(M), size=16)
    
#f3.tight_layout()        
f3.show()
