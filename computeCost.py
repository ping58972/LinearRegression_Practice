import numpy as np
def computeCost(X, y, theta):
# """ COMPUTECOST(X, y, theta) computes the cost of using theta as the
#     parameter for linear regression to fit the data points in X and y """ 
    m = len(y)
    h = np.dot(X, theta)
    J = np.subtract(h, y)
    J = np.power(J,2)
    J = np.sum(J)
    J = (1/(2*m))*J
    return J

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

    J = computeCost(X,y,theta)
    print(J)
    J = computeCost(X,y,[[-1],[2]])
    print(J)