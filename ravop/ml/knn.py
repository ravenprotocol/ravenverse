import numpy as np
import logging
import logging.handlers
from ravop import globals as g
from ravop.core import Graph, Tensor, Scalar, square_root, argmax
# KNN Regression

class KNN(Graph):

    print(" \n ----------------------- KNN OBJECT INSTANTIATED --- GOOD TO GO ---------------------- \n")

    def __init__(self, X_train, y_train, id = None,  **kwargs):
        """ 
        
        Called as soon as the object of KNN class is created.

        Parameters:

                    X_train = Input Training Data without the Label(Target)
                    y_train = Target Label from Training Data
                    n_neighbours = Number of Neighbours to be considered in KNN. Default Value is 5.
                    weights = Weights assigned to the distance based calculation. Default Value is "uniform"


        """

        super().__init__(id = id, **kwargs)
        self.__setup_logger()
        # defining hyperparameters
        # X_train = args.get("X_train", None)
        # self.X_train = Tensor(X_train, name = "X_train")
        self.X_train = Tensor(X_train, name = "X_train")

        self.y_train = Tensor(y_train, name = "y_train")
        #
        # y_train = args.get("y_train", None)
        # self.y_train = Tensor(y_train, name = "y_train")

        self.n_neighbours = kwargs.get("n_neighbours", None)
        if self.n_neighbours is None:
            self.n_neighbours = 5

        self.weights = kwargs.get("weights", None)
        if self.weights is None:
            self.weights = "uniform"

        self.n_classes = kwargs.get("n_classes", None)
        if self.n_classes is None:
            self.n_classes = 3

    def __setup_logger(self):

        # Set up a specific logger with our desired output level
        self.logger = logging.getLogger(KNN.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        # Add the log message handler to the logger
        handler = logging.handlers.RotatingFileHandler(g.ravop_log_file)

        self.logger.addHandler(handler)


    def euclidean_distance(self, a, b):
        """ 
        
        Returns a scalar Euclidean Distance value between two points on a 2-D plane 
        
        Parameters:

                    a = Point_1 on the plane
                    b = Point_2 on the plane

        Output:

                Scalar Value for Distance between the two points.
        
        """
        a = Tensor(a, name = "a")
        while square_root(sum((a.sub(b)).pow(Scalar(2)), axis=1)).status == "pending":
            pass
        # np.sqrt(sum((a-b)**2), axis = 1)
        print("\n\n Final Status \n\n", square_root(sum((a.sub(b)).pow(Scalar(2)), axis=1)).status)
        return square_root(sum((a.sub(b)).pow(Scalar(2)), axis=1)).output

    
    def KNN_neighbours(self, X_test, return_distance = False):
        """ 
        
        Returns the N nearest Neighbours based on Euclidean Distance, specifically indexes.

        Parameters:

                    X_test = Test Data
                    return_distance = if True then distance values will be returned

        Output:
                Returns N Nearest Neighbours indexes
        """

        distance = []
        # distance values b/w points
        neighbour_index = []
        # indexes of neighbours

        point_distance = [self.euclidean_distance(x_test, self.X_train) for x_test in X_test.output]
        # return is Scalar/Tensor
        # each row has a list distance value between test point 1 and all the individual training data points.

        for i in point_distance:

            # enumerate so as to preserve index and value
            enumerated_neighbour = enumerate(i)
            # sorted list of N nearest neighbours
            sorted_neighbour = sorted(enumerated_neighbour, key = lambda x: Scalar(x[1]))[:self.n_neighbours]

            # index list for sorted N nearest neighbours
            index_list = [Scalar(t[0]) for t in sorted_neighbour]
            # distance value list for sorted N nearest neighbours
            distance_list = [Scalar(t[1]) for t in sorted_neighbour]

            # appending
            distance.append(distance_list)
            neighbour_index.append(index_list)

        if return_distance:
            return np.array(distance), np.array(neighbour_index)
        
        return np.array(neighbour_index)


    def predict(self, X_test):
        """ 
        predict the data

        Parameters:
                    X_test = Data on which prediction has to be made

        Output:
                Gives you the Prediction
        
        """
        if self.weights == "uniform":
            neighbours = self.KNN_neighbours(X_test)
            # to understand bincount(), visit - https://i.stack.imgur.com/yAwym.png
            y_pred = np.array([argmax(np.bincount(self.y_train[neighbour])) for neighbour in neighbours])

            return y_pred

        if self.weights == "distance":

            # N nearest neighbours distance and indexes
            distance, neighbour_index = self.KNN_neighbours(X_test, return_distance = True)
            
            inverse_distance = Scalar(1).div(distance)

            mean_inverse_distance = inverse_distance.div(sum(inverse_distance, axis=1)[:, np.newaxis])

            proba = []

            # running loop on K nearest neighbours elements only and selecting train for them
            for i , row in enumerate(mean_inverse_distance):

                row_pred = self.y_train[neighbour_index[i]]

                for k in range(self.n_classes):
                    indices = np.where(row_pred.equal(k))
                    prob_ind = sum(row[indices])
                    proba.append(np.array(prob_ind))

            predict_proba = np.array(proba).reshape(Scalar(X_test.shape[0]), self.n_classes)
            y_pred = np.array([argmax(item) for item in predict_proba])
            
            return y_pred


    def score(self, X_test, y_test):
        """ 
        Used to measure performance of our algorithm

        Parameters:
                    X_test = Test data
                    y_test = Target Test Data

        Output:
                Returns the Score Value
        """
        X_test = Tensor(X_test, name = "X_test")
        y_test = Tensor(y_test, name = "y_test")
        y_pred = Tensor(self.predict(X_test), name = "y_pred")
        
        return float(Scalar(sum(y_pred.equal(y_test)))) / float(Scalar(len(y_test)))


    

    