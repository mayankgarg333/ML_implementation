# Using regularization 
#This particular choice of regularizer is known in the machine learning literature as
#weight decay because in sequential learning algorithms, it encourages weight values
#to decay towards zero, unless supported by the data. In statistics, it provides an example
#of a parameter shrinkage method because it shrinks parameter values towards zero
# Bisop page 144


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
## USing the linear regression model given at page 144 Bisop
## X: defined as the matrix of all observation and all features (all input data)
## Each roq of X is a data point. So for any data point estimated y will be W' * X' (row)
## X will have the dimension of: N*m (number of row are data point, and features are column)
## W: weight vector of m entries. 
## W= inv(X' X +lambda*I) * X' t ; t is actual data points with noise 
## the equation is calculated using error minimization, error function have lambda term for regularization
####

# create X:

M=5;                            # CHANGE THIS TO CHECK THE EFFECT OF DIFFERENT M
lmda=np.exp(1)                  # CHANGE THIS TO CHECK THE EFFECT OF DIFFERENT lambda values
X=np.array([])
for i in range(M):
    X=np.vstack([X,x**i]) if X.size else x**i    # use of vstack and hstack for matlab type manuplation 
X=X.T
W=np.dot(np.linalg.pinv(lmda*np.identity(M)+np.dot(X.T,X)),np.dot(X.T,t))
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
## Effect of M value (Effect of the order of the polynomial to fit), fixed lambda
###

# Y will hold the prediction for each M value
Y=np.array([])
for M in range(1,11):
    X=np.array([])
    for i in range(M):
        X=np.vstack([X,x**i]) if X.size else x**i    # use of vstack and hstack for matlab type manuplation 
    X=X.T
    I=np.linalg.pinv(lmda*np.identity(M)+np.dot(X.T,X)) if np.dot(X.T,X).shape else 1/np.dot(X.T,X)
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




###
## Effect of lambda for M=5 and M=9; first M=5
###
M=5
L=np.linspace(-5,10,10)
Y=np.array([])
for i in range(1,11):
    lmda=np.exp(L[i-1])
    X=np.array([])
    for i in range(M):
        X=np.vstack([X,x**i]) if X.size else x**i    # use of vstack and hstack for matlab type manuplation 
    X=X.T
    I=np.linalg.pinv(lmda*np.identity(M)+np.dot(X.T,X)) if np.dot(X.T,X).shape else 1/np.dot(X.T,X)
    W=np.dot(I,np.dot(X.T,t))
    y_pred=np.dot(W.T,X.T)
    Y=np.vstack([Y,y_pred]) if Y.size else y_pred
    
    
f4=plt.figure(4, figsize=(20,20))
for i in range(1,11):
    
    p=plt.subplot2grid((2,5), ((i-1)/5,(i-1)%5))
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
    s = "%.2f" % L[i-1]
    p.set_title("lambda: exp("+s+")", size=16)
    
#f3.tight_layout()   
f4.suptitle("M: 5", size=16)  
f4.show()

###
## Effect of lambda for M=9
###

M=9
L=np.linspace(-20,20,10)
Y=np.array([])
for i in range(1,11):
    lmda=np.exp(L[i-1])
    X=np.array([])
    for i in range(M):
        X=np.vstack([X,x**i]) if X.size else x**i    # use of vstack and hstack for matlab type manuplation 
    X=X.T
    I=np.linalg.pinv(lmda*np.identity(M)+np.dot(X.T,X)) if np.dot(X.T,X).shape else 1/np.dot(X.T,X)
    W=np.dot(I,np.dot(X.T,t))
    y_pred=np.dot(W.T,X.T)
    Y=np.vstack([Y,y_pred]) if Y.size else y_pred
    
    
f5=plt.figure(5, figsize=(20,20))
for i in range(1,11):
    
    p=plt.subplot2grid((2,5), ((i-1)/5,(i-1)%5))
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
    s = "%.2f" % L[i-1]
    p.set_title("lambda: exp("+s+")", size=16)
    
#f3.tight_layout()   
f5.suptitle("M: 8", size=16)  
f5.show()