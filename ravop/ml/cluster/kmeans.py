import ravop.core as R
from ravop.core import Graph, Tensor, Scalar, square_root,add, min
from ravop.utils import inform_server
import matplotlib.pyplot as plt


class Kmeans(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.points = None
        self.label = None
        self.centroids=None
        self.k = None

    def set_params(self, **kwargs):
        self.params.update(**kwargs)

    def get_params(self):
        return self.params

    def fit(self, X, k=3, iter=10):
        self.points = R.Tensor(X)
        self.k = k
        self.centroids = self.initialize_centroids()
        inform_server()
        self.label = self.closest_centroids(self.centroids)
        self.update_centroids()

        for i in range(iter):
            print('iteration',i)
            self.update_centroids()
            self.label = self.closest_centroids(self.centroids)
            inform_server()
        while self.label.status!="computed":
            pass

    def initialize_centroids(self):
        return R.random(self.points, size=self.k)

    def closest_centroids(self, centroids):
        centroids = R.expand_dims(centroids, axis=1)
        return R.argmin(square_root(R.sub(self.points, centroids).pow(Scalar(2)).sum(axis=2)))


    def update_centroids(self):

        gather = R.gather(self.points, R.find_indices(self.label, values=[0])).mean(axis=1)
        for i in range(1, self.k):
            ind = R.find_indices(self.label, values=[i])
            gat = R.gather(self.points, ind).mean(axis=1)
            gather = R.concat(gather, gat)
        self.centroids= gather.reshape(shape=[self.k, len(self.points.output[0])])
        inform_server()
    def plot(self):
        fig, axs = plt.subplots(1)
        axs.scatter(self.points.output[:, 0], self.points.output[:, 1], c=self.label.output)
        #axs.scatter(self.centroids.output[:, 0], self.centroids.output[:, 1] ,'X', color="black",markersize=10)
        plt.show()



class MiniBatchKmeans(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.X=None
        self.points = None
        self.label = None
        self.centroids = None
        self.batch_size=None
        self.k = None

    def initialize_centroids(self):
        cen=R.random(self.points, size=self.k)
        while cen.status!='computed':
            pass
        inform_server()
        print(cen)
        return cen

    def Mini_batch(self,points,batch_size):
        mb=R.random(points,size=batch_size)
        return mb

    def closest_centroids(self,points,centroids):
        centroids = R.expand_dims(centroids, axis=1)
        return R.argmin(R.square_root(R.sum(R.square(R.sub(points, centroids)), axis=2)))


    def update_centroids(self,points,label):
        while label.status!= 'computed':
            pass
        if 0 in label.output :
            gather=R.gather(points,R.find_indices(label,values=[0])).mean(axis=1)
        else:
            gather=R.gather(self.centroids, Tensor([0])).expand_dims(axis=0)

        for i in range(1,self.k):
            if i in label.output:
                ind = R.find_indices(label,values=[i])
                gat = R.gather(points,ind).mean(axis=1)
            else:
                gat = R.gather(self.centroids, Tensor([i])).expand_dims(axis=0)
            gather=R.concat(gather,gat)

            while gat.status!='computed':
                pass
        return gather.reshape(shape=[self.k,len(self.points.output[0])])


    def fit(self, X, k , iter=5,batch_size=None):
        inform_server()
        self.points=Tensor(X)
        self.k=k
        self.iter=iter
        self.batch_size=batch_size
        self.centroids=self.initialize_centroids()
        #self.label=self.closest_centroids(self.points,self.centroids)
        points=self.Mini_batch(self.points,batch_size=batch_size)
        label=self.closest_centroids(points,self.centroids)
        print(3)
        self.centroids=self.update_centroids(points,label)
        inform_server()
        for i in range(iter):
            print('iteration',i)
            points = self.Mini_batch(self.points, batch_size=self.batch_size)
            label = self.closest_centroids(points, self.centroids)
            self.centroids=self.update_centroids(points,label)

            inform_server()

        self.label = self.closest_centroids(self.points, self.centroids)
        while self.label.status!="computed":
            pass
        return self.label




if __name__ == '__main__':
    pass

    k.plot()
    #k.fit(X, k=3,iter=300)
    #k.plot()
