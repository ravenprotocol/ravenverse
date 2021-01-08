import numpy as np
import logging
import logging.handlers
from ravop import globals as g
from ravop.core import Graph, Tensor, Scalar, square_root, argmax
import sys
import time
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

        y_train = y_train.reshape(len(y_train),1)
        self.y_train = Tensor(y_train, name = "y_train")
        print("\n Shape of y_train is \n", y_train.shape)
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
        sq_cal = square_root(((a.sub(b)).pow(Scalar(2))).sum(axis=1))
        while sq_cal.status != "computed":
            pass
        # np.sqrt(sum((a-b)**2), axis = 1)
        return sq_cal.output

    
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
        print(point_distance, "\n point-distance list", type(point_distance), "data type")
        # N-d list

        # each row has a list distance value between test point 1 and all the individual training data points.

        for i in point_distance:

            # enumerate so as to preserve index and value
            enumerated_neighbour = enumerate(i)
            print(enumerated_neighbour, "\n enumerated_neighbour", type(enumerated_neighbour), "data type")
            # sys.exit(0)
            # sorted list of N nearest neighbours
            sorted_neighbour = sorted(enumerated_neighbour, key = lambda x: x[1])[:self.n_neighbours]
            print(sorted_neighbour, "\n sorted_neighbor", type(sorted_neighbour), "data type")
            # 2 d list, with N nearest entries only like this [[a, b, c], [d, e, f], ...]

            # index list for sorted N nearest neighbours
            index_list = [t[0] for t in sorted_neighbour]
            print(index_list, "\n index_list \n", type(index_list), "data type")
            # distance value list for sorted N nearest neighbours
            distance_list = [t[1] for t in sorted_neighbour]
            print(distance_list, "\n distance_list \n", type(distance_list), "data type")

            # appending
            distance.append(distance_list)
            neighbour_index.append(index_list)
            print(distance, "\n distance \n", type(distance), "data type of distance", neighbour_index,
                  "\n neighbour_index \n", type(neighbour_index), "data type")

        if return_distance:
            distance = Tensor(distance, name = "distance")
            print(distance, "\n distance tensor created")
            neighbour_index = Tensor(neighbour_index, name = "neighbour_index")
            print(neighbour_index, "\n neighbour index tensor created")

            while distance.status and neighbour_index.status != "computed":
                pass
            print(distance.output, "\n distance output when return distance is true \n",neighbour_index.output, "\n neighbour_index output when it is true \n")
            return distance.output, neighbour_index.output
        
        neighbour_index = Tensor(neighbour_index, name = "neighbour_index_outer")
        while neighbour_index.status != "computed":
            pass
        print(neighbour_index.output, "\n neigbour_index from function when it is False\n")
        return neighbour_index.output


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
            print(neighbours, "\n neighbours \n", type(neighbours), "data type")
            # neighbours is a Tensor, use neighbours.output for converting to nd array
            # to understand bincount(), visit - https://i.stack.imgur.com/yAwym.png
            y_pred = Tensor([argmax(np.bincount(self.y_train[neighbour])) for neighbour in neighbours.output], name = "y_pred from uniform weights")
            print(y_pred, "\n y_pred \n", type(y_pred), "data type")

            return y_pred

        if self.weights == "distance":

            # N nearest neighbours distance and indexes
            distance, neighbour_index = self.KNN_neighbours(X_test, return_distance = True)
            print(distance, "\n distance", neighbour_index, "\n neighbour_index", type(distance), "distance type",
                  type(neighbour_index), "neighbour_index data type")
            distance = Tensor(distance, name = "distance_in_inverse")
            print(distance, "\n distance tensor for inverse distance calculation \n")
            # from here it does not work..

            inverse_distance = Scalar(1).div(distance)
            while inverse_distance.status != "computed":
                pass
            print("\n inverse_distance_first created \n", inverse_distance)

            mean_inverse_distance = inverse_distance.div(inverse_distance.sum(axis=1).output[:, np.newaxis])
            while mean_inverse_distance.status != "computed":
                pass

            print(mean_inverse_distance, "\n mean_inverse_distance", type(mean_inverse_distance),
                  "data type of mean_inverse_distance")


            mean_inverse_distance = Tensor(mean_inverse_distance, name="mean_inverse_distance")

            proba = []

            # running loop on K nearest neighbours elements only and selecting train for them
            for i , row in enumerate(mean_inverse_distance.output):

                row_pred = self.y_train[neighbour_index.output[i]]
                print(row_pred, "\n row_pred", type(row_pred), "data type")

                for k in range(self.n_classes):
                    indices = np.where((Tensor(row_pred, name = "row_pred").equal(k)).output)
                    while indices.status !="computed":
                        pass
                    print(indices, "\n indices", type(indices), "data type")
                    prob_ind = sum(row[indices])
                    print(prob_ind, "\n prob_ind", type(prob_ind), "data type")
                    proba.append(Tensor(prob_ind, name = "prob_ind").output)
                    print(proba, "proba")

            predict_proba = Tensor(proba, name = "proba").reshape(Scalar(X_test.shape[0]), self.n_classes)
            print(predict_proba, "predict_proba", type(predict_proba), "data type")
            y_pred = Tensor([argmax(Scalar(item)) for item in predict_proba.output], name = "y_pred")
            print(y_pred, "y_pred", type(y_pred), "data type")
            
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
        y_test = y_test.reshape(len(y_test), 1)
        y_test = Tensor(y_test, name = "y_test")
        print("\n Shape of y_test \n", y_test.shape)
        y_pred = Tensor(self.predict(X_test), name = "y_pred")
        
        return float(Scalar(sum(y_pred.equal(y_test)))) / float(Scalar(len(y_test)))