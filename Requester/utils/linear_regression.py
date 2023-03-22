from dotenv import load_dotenv
load_dotenv()

import numpy as np
import ravop as R

import matplotlib.pyplot as plt

def r2_score(y_true, y_pred):
    if isinstance(y_true, R.Tensor) or isinstance(y_true, R.Op):
        pass
    else:
        y_true = R.Tensor(y_true, name="y_true")

    if isinstance(y_pred, R.Tensor) or isinstance(y_pred, R.Op):
        pass
    else:
        y_pred = R.Tensor(y_pred, name="y_pred")

    print(type(y_true), type(y_pred))

    scalar1 = R.t(1)

    SS_res = R.sum(R.square(R.sub(y_pred, y_true)))
    SS_tot = R.sum(R.square(R.sub(y_true, R.mean(y_true))))

    return R.sub(scalar1, R.div(SS_res, SS_tot))

class LinearRegression():
    def __init__(self, x_points, y_points, theta):
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
        for e in range(1, num_iters + 1):
            residual = self.X.dot(self.theta).sub(self.y)
            temp = self.theta.sub((alpha_.div(self.m)).multiply(self.X.transpose().dot(residual)))
            print('Iteration : ', e)
            self.theta = temp
        self.op_theta = self.theta
        self.theta.persist_op(name="theta")

    def predict(self, X_test):
        xt = X_test
        if not isinstance(X_test, R.Op):
            xt = R.t(X_test)

        y_val = xt.dot(self.op_theta).squeeze()
        y_val.persist_op(name="predicted values")
        return y_val

    def score(self, X, y, name="r2"):
        if not isinstance(X, R.Tensor):
            X = R.t(X)
        if not isinstance(y, R.Tensor):
            y = R.t(y)

        y_pred = self.predict(X)
        y_true = y

        if name == "r2":
            score = r2_score(y_true, y_pred)
        else:
            return None
        score.persist_op(name="score")

    def plot_graph(self, optimal_theta):
        fig, ax = plt.subplots()
        ax.plot(self.raw_X[:, 1], self.raw_y[:, 0], 'o', label='Raw Data')
        ax.plot(self.raw_X[:, 1], self.raw_X.dot(optimal_theta), linestyle='-', label='Linear Regression')

        plt.show()

    def set_params(self, **kwargs):
        param_dict = {
            'theta': self.theta(),
            'X': self.X(),
            'y': self.y()
        }
        for i in kwargs.keys():
            if i in param_dict.keys():
                param_dict[i] = kwargs[i]

        return param_dict

    def get_params(self):
        param_dict = {
            'theta': self.theta(),
            'X': self.X(),
            'y': self.y()
        }
        return param_dict

def preprocess(data):
    x = data[:, 0]
    y = data[:, 1]
    y = y.reshape(y.shape[0], 1)
    x = np.c_[np.ones(x.shape[0]), x]
    theta = np.zeros((2, 1))
    return x, y, theta

def get_dataset():
    data = np.loadtxt('examples/datasets/data_linreg.txt', delimiter=',')
    x, y, theta = preprocess(data)
    return x, y, theta

def create_model(x, y, theta):
    model = LinearRegression(x, y, theta)
    return model

def train(model, alpha=0.01, iterations=5):
    model.compute_cost()
    model.gradient_descent(alpha, iterations)
    return model

def test(model, x_test, y_test):
    model.predict(x_test)
    model.score(x_test, y_test)

def compile():
    R.activate()

def execute(participants=1):
    R.execute(participants=participants)
    R.track_progress()
