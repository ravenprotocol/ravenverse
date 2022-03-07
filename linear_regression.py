import matplotlib.pyplot as plt
import ravop as R
import numpy as np
import pathlib

algo = R.Graph(name='lin_reg', algorithm='linear_regression', approach='distributed')

class LinearRegression():
    def __init__(self,x_points,y_points,theta):
        self.raw_X = x_points
        self.raw_y = y_points
        self.m = R.t(self.raw_y.shape[0])
        self.X = R.t(self.raw_X.tolist())
        self.y = R.t(self.raw_y.tolist())
        self.theta = R.t(theta.tolist())

    def compute_cost(self):
        residual = self.X.dot(self.theta).sub(self.y)
        return (R.t(1).div(R.t(2).multiply(self.m))).multiply(residual.dot(residual.transpose()))

    def gradient_descent(self, alpha, num_iters):
        alpha_ = R.t(alpha)
        for e in range(1,num_iters+1):
            residual = self.X.dot(self.theta).sub(self.y)
            temp = self.theta.sub((alpha_.div(self.m)).multiply(self.X.transpose().dot(residual)))
            self.theta = R.t(temp())
            print('Iteration : ',e)
        op_theta = self.theta()
        print('Theta found by gradient descent: intercept={0}, slope={1}'.format(op_theta[0],op_theta[1]))
        return self.theta, op_theta[0], op_theta[1]

    def plot_graph(self,optimal_theta, res_file_path):
        optimal_theta = optimal_theta()
        fig, ax = plt.subplots()
        ax.plot(self.raw_X[:,1], self.raw_y[:,0], 'o', label='Raw Data')
        ax.plot(self.raw_X[:,1], self.raw_X.dot(optimal_theta), linestyle='-', label='Linear Regression')
        plt.ylabel('Profit')
        plt.xlabel('Population of City')
        legend = ax.legend(loc='upper center', shadow=True)
        plt.savefig(res_file_path)
        plt.show()

def preprocess(data):
    x = data[:,0]
    y = data[:,1]
    y = y.reshape(y.shape[0], 1)
    x = np.c_[np.ones(x.shape[0]), x] # adding column of ones to X to account for theta_0 (the intercept)
    theta = np.zeros((2, 1))
    return x,y,theta

iterations = 5
alpha = 0.01

data = np.loadtxt('data/data_linreg.txt', delimiter=',')

x,y,theta = preprocess(data)

model = LinearRegression(x,y,theta)
model.compute_cost()            # initial cost with coefficients at zero
optimal_theta, inter, slope = model.gradient_descent(alpha, iterations)
print(optimal_theta, inter, slope)
res_file_path = str(pathlib.Path().resolve()) + '/result.png'
print(res_file_path)
model.plot_graph(optimal_theta, res_file_path)

algo.end()