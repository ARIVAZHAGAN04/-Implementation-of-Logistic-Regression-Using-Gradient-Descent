# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1: Use the standard libraries in python for finding linear regression.

STEP 2: Set variables for assigning dataset values

STEP 3: Import linear regression from sklearn.

STEP 4: Predict the values of array

STEP 5: Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

## Program:
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Arivazhagan G R
RegisterNumber:  212223040020
*/
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data=np.loadtxt("ex2data1.txt",delimiter=',')
x=data[:,[0,1]]
y=data[:,2]
x[:5]
y[:5]
plt.figure()
plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted")
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()
def sigmoid(z):
  return 1/(1+np.exp(-z))
  plt.plot()
x_plot=np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot))
plt.show()
def costfunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(X.T,h-y)/X.shape[0]
  return J,grad
  x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
j,grad = costfunction(theta,x_train,y)
print(j)
print(grad)
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad = costfunction(theta,x_train,y)
print(j)
print(grad)
def plotDecisionBoundary(theta,x,y):
  x_min,x_max=x[:,0].min() -1,x[:,0].max()+1
  y_min,y_max=x[:,1].min() -1,x[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot=np.c_[xx.ravel(),yy.ravel()]
  x_plot=np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot=np.dot(x_plot,theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted")
  plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted")
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return J
  def gradient(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  grad=np.dot(x.T,h-y)/x.shape[0]
  return grad
  x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)
```
## Output:
![logistic regression using gradient descent](sam.png)

![198872325-63b06458-0515-4f3e-b324-c4323abd829f](https://github.com/ARIVAZHAGAN04/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/161414455/1c615923-d2e4-4f26-a22a-78159f82c34b)

![198872334-d5651b58-e5cf-4a24-8420-86428198962a](https://github.com/ARIVAZHAGAN04/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/161414455/43f19d4a-8f15-4b09-ab8c-6b41927759b6)

![198872345-49603ced-7704-475c-976b-17529b00c789](https://github.com/ARIVAZHAGAN04/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/161414455/93f86a80-830b-4a8c-a82f-5fa61a7725e3)

![198872352-e2d984bc-9acd-491a-8a77-3433ff213379](https://github.com/ARIVAZHAGAN04/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/161414455/5a0227e0-66f3-4666-9a33-b66083575f87)





## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

