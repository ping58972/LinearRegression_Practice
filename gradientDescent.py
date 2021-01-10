import numpy as np

from computeCost import *

def gradientDescent(X, y, theta, alpha, num_iters):
#     """
#     GRADIENTDESCENT Performs gradient descent to learn theta
#    theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
#    taking num_iters gradient steps with learning rate alpha
#     """
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        temp0 = theta[0] - (alpha * (1/m) * np.sum(np.subtract(np.dot(X, theta), y)))
        temp1 = theta[1] - (alpha * (1/m) * np.sum(np.multiply(np.subtract(np.dot(X, theta), y),X[:,1].reshape(m,1))))
        theta[0] = temp0
        theta[1] = temp1
        
        J_history[i] = computeCost(X, y, theta)
        
    return (theta, J_history)  

if __name__ == "__main__":
    data = np.loadtxt("data/ex1data1.txt", delimiter=',')
    X = data[:,0]
    y = data[:, 1]
    m = len(y)

    X = np.hstack((np.ones((m,1)),X.reshape(m,1)))
    y = y.reshape(m,1)
    theta = np.zeros((2,1))
    iterations = 1500
    alpha = 0.01

    theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
    print(theta)