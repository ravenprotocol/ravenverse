class KMeans(object):
    def __init__(self, **kwargs):
        self.params = kwargs

    def set_params(self, **kwargs):
        self.params.update(**kwargs)

    def get_params(self):
        return self.params

    def fit(self, X, k=3, iter=10):
        """
        1. Random Initialization of centroids
        2.
        """





if __name__ == '__main__':
    k = KMeans()
    k.set_params(**{"a": "b"})
    print(k.get_params())
