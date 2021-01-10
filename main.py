import numpy as np
from pylab import plot
# %matplotlib inline
from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')

from computeCost import *
from gradientDescent import *

data = np.loadtxt("data/ex1data1.txt", delimiter=',')

X = data[:,0]
y = data[:, 1]
m = len(y)

# print(X)
plt.plot(X,y, 'r+', color='black')
plt.show()


X = np.hstack((np.ones((m,1)),X.reshape(m,1)))
y = y.reshape(m,1)
theta = np.zeros((2,1))
iterations = 1500
alpha = 0.01

J = computeCost(X,y,theta)
print(J)
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
print(theta)

plt.plot(X[:,1],y, 'r+')
plt.plot(X[:,1], np.dot(X,theta), color='black')
plt.show()

predict1 = [1,3.5] @ theta
predict2 = [1,7] @ theta
print(predict1, predict2)