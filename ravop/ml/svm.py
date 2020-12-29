# importing packages
import numpy as np  # for handling multi-dimensional array operation
import pandas as pd  # for reading data from csv 
import statsmodels.api as sm  # for finding the p-value
from sklearn.preprocessing import MinMaxScaler  # for normalization
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
import numpy as np
import logging
import logging.handlers
from ravop import globals as g
from ravop.core import Graph, Tensor, Scalar, square_root, argmax
import sys



class Support_Vector_Machine(Graph):

    print("\n ------------------ SVM OBJECT INITIATED ------------------ \n")

    
    def __init__(self, id = None, **kwargs):
        # regularisation_parameter = 10000, learning_rate = 0.000001

        super().__init__(id = id, **kwargs)
        self.__setup_logger()
        self.regularisation_parameter = kwargs.get("regularisation_parameter", None)
        if self.regularisation_parameter is None:
            self.regularisation_parameter = 10000

        self.learning_rate = kwargs.get("learning_rate", None)
        if self.learning_rate is None:
            self.learning_rate = 0.000001

    def __setup_logger(self):

        # Set up a specific logger with our desired output level
        self.logger = logging.getLogger(Support_Vector_Machine.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        # Add the log message handler to the logger
        handler = logging.handlers.RotatingFileHandler(g.ravop_log_file)

        self.logger.addHandler(handler)

        
    print("\n\n--------------- SVM MODEL STARTS ---------------\n\n")
    
    
    def computing_cost(self, W, X, Y):
        """

            It will calculate the optimal parameters for W and b parameters in order to minimise the cost function.

            Parameters:
                        W = Weights
                        X = Input Features
                        Y = Target Output


            Output:
                It returns the cost
        
        """
        W = Tensor(W, name = "W")
        X = Tensor(X, name = "X")
        Y = Tensor(Y, name = "Y")

        N = X.shape[0]
        distances = Scalar(1).sub((Y.matmul(X.dot(W))))
        # distances = 1 - Y*(np.dot(X, W))
        # max(0, distance)
        distances[distances.less(Scalar(0))] = Scalar(0)
        loss = Scalar(self.regularisation_parameter).mul(sum(distances)/N)
        # find cost
        cost = Scalar(0.5).mul((W.dot(W))).add(loss)
    
        return cost

    def calculate_cost_gradient(self, W, X_batch, Y_batch):
        """
        
        Calculating Cost for Gradient

        Parameters:
                    X_batch = Input features in batch or likewise depending on the type of gradient descent method used
                    Y_batch = Target features in batch or likewise depending on the type of gradient descent method used

        Output:
                Weights Derivatives

        """
        
        if type(Y_batch) == np.float64:
            Y_batch = np.array([Y_batch])
            X_batch = np.array([X_batch])
            
        distance = 1 - (Y_batch * np.dot(X_batch, W))
        dw = np.zeros(len(W))
        
        for ind, d in enumerate(distance):
            
            if max(0, d) == 0:
                di = W
                
            else:
                di = W - (self.regularisation_parameter * Y_batch[ind] * X_batch[ind])
            
            dw += di
            
        dw = dw/len(Y_batch)  # average
        
        return dw


    def Stochastic_gradient_descent(self, features, outputs):
        
        """
        
        SGD to calculate Gradients such that only a Single points are considered to update weights

        Parameters:
                    features = Input Features
                    outputs = outputs

        Output:
                weights


        """

        
        max_epochs = 5000
        weights = np.zeros(features.shape[1])
        print("\n\n----------------- STOCHASTIC GRADIENT DESCENT RUNNING -----------------\n\n")

        nth = 0
        prev_cost = float("inf")
        print("\n Previous Cost:", prev_cost)
        cost_threshold = 0.01  # in percent
        # stochastic gradient descent
        for epoch in range(1, max_epochs):
            # shuffle to prevent repeating update cycles
            X, Y = shuffle(features, outputs)
            for ind, x in enumerate(X):
                ascent = self.calculate_cost_gradient(weights, x, Y[ind])
                weights = weights - (self.learning_rate * ascent)

            # convergence check on 2^nth epoch
            if epoch == 2 ** nth or epoch == max_epochs - 1:
                cost = self.computing_cost(weights, features, outputs)
                print("Epoch is: {} and Cost is: {}".format(epoch, cost))
                # stoppage criterion
                if abs(prev_cost - cost) < cost_threshold * prev_cost:
                    return weights
                prev_cost = cost
                nth += 1
        return weights

    def remove_correlated_features(self, X):
        """
        Removing Correlated Fetures
        Parameters:
                    X = input variables
                
        Output:
                Removing correlated features

        """

        corr_threshold = 0.9
        corr = X.corr()
        drop_columns = np.full(corr.shape[0], False, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[0]):
                if corr.iloc[i, j] >= corr_threshold:
                    drop_columns[j] = True
                    
        columns_dropped = X.columns[drop_columns]
        X.drop(columns_dropped, axis=1, inplace=True)
        
        return columns_dropped


    def remove_less_significant_features(self, X, Y):
        """
        Removing Less significant features

        Parameters:
                    X=input features
                    Y=output
        Output:
                Less important features removed
        """
        
        sl = 0.05
        regression_ols = None
        columns_dropped = np.array([])
        for i in range(0, len(X.columns)):
            
            regression_ols = sm.OLS(Y, X).fit()
            max_col = regression_ols.pvalues.idxmax()
            max_val = regression_ols.pvalues.max()
            
            if max_val > sl:
                X.drop(max_col, axis='columns', inplace=True)
                columns_dropped = np.append(columns_dropped, [max_col])
            else:
                break
        regression_ols.summary()
        
        return columns_dropped

    def train_svm(self, X, Y, X_train, X_test, y_train, y_test):
        """
        Training SVM

        Parameters:
                    X=input features
                    Y=output class
                    X_train = training input features
                    X_test = testing input features
                    y_train = training output
                    y_test = testing output

        Output:
                Trained Weights

        """
        
        # insert 1 in every row for intercept b
        X_train.insert(loc=len(X_train.columns), column='intercept', value=1)
        X_test.insert(loc=len(X_test.columns), column='intercept', value=1)

        # train the model
        print("***** TRAINING IS STARTED ****")
        W = self.Stochastic_gradient_descent(X_train.to_numpy(), y_train.to_numpy())
        # above operation's aim is to return us the optimised weights for OPTIMISATION problem.
        print("**** TRAINING COMPLETED ****")
        print("WEIGHTS ARE AS FOLLOWS: {}".format(W))

        return X, Y, X_train, X_test, y_train, y_test, W

    def test_svm(self, X, Y, X_train, X_test, y_train, y_test, W):
        """
        Testing/Predict SVM

        Parameters:
                    X=input features
                    Y=output class
                    X_train = training input features
                    X_test = testing input features
                    y_train = training output
                    y_test = testing output
                    W=Weights trained

        Output:
                y_test_predicted = Predictions Made

        """
        

        print("**** TEST THE MODEL ****")

        y_train_predicted = np.array([])
        for i in range(X_train.shape[0]):
            yp = np.sign(np.dot(X_train.to_numpy()[i], W))
            y_train_predicted = np.append(y_train_predicted, yp)

        y_test_predicted = np.array([])
        for i in range(X_test.shape[0]):
            yp = np.sign(np.dot(X_test.to_numpy()[i], W))
            y_test_predicted = np.append(y_test_predicted, yp)

        print("accuracy on test dataset: {}".format(accuracy_score(y_test, y_test_predicted)))
        print("recall on test dataset: {}".format(recall_score(y_test, y_test_predicted)))
        print("precision on test dataset: {}".format(recall_score(y_test, y_test_predicted)))

        return y_test_predicted




