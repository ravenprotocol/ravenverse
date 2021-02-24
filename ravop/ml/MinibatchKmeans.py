import numpy as np
import ravop.core as R
from ravop.core import Tensor
'''
                
                    ****    Using Ravop   ****
                
'''
''' 
class MinibatchKmeans():

    def __init__(self,**kwargs):
        self.params = kwargs
        self.points = None
        self.label = None
        self.b=None
        self.k = None

    def set_params(self, **kwargs):
        self.params.update(**kwargs)

    def get_params(self):
        return self.params

    def fit(self,X,k,b,iter=100):
        self.points= X
        self.k=k
        self.iter=iter
        self.centroids=self.initialize_centroiods()

        pass

    def initialize_centroiods(self):
        return R.random(self.points, size=self.k)

    def closest_centroids(self,centroids):
        centroids = R.expand_dims(centroids, axis=1)
        return R.argmin(R.square_root(R.sum(R.square(R.sub(self.points, centroids)), axis=2)))

    def minibatch(self):
        pass
'''




'''

            ****  Using numpy ****

'''

class MinibatchKmeans():
    def __init__(self,points,k,batchsize,max_iter=300):
        self.points=points
        self.k=k
        self.b= batchsize
        self.iter=max_iter
        self.centroids = self.initialize_centroids(points, k)
        print(self.centroids)
        self.label = self.closest_centroid(points, self.centroids)

        self.temp=self.label

    def initialize_centroids(self,points,k):
        cent=np.array(points.copy())
        np.random.shuffle(cent)
        return cent[:k]

    def closest_centroid(self,points, centroids):
        distances = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        print(distances)
        return np.argmin(distances, axis=0)

    def update_centroids(self):
        newcen=[]
        for k in range(self.k):
            p=self.points[self.label==k].mean(axis=0)
            newcen.append(p)

        return np.array(newcen)

    def fit(self):
        #update
        for i in range(self.iter):
            self.centroids = self.update_centroids()
            self.label = self.closest_centroid(self.minibatch(self.points,7), self.centroids)
            pass
        print(self.centroids)
        return self.label

    def minibatch(self,points,b):
        '''
            returns the minibatch for centroid computation
        '''
        x=points
        np.random.shuffle(x)
        return x[:b]
        pass

points=np.array([[1,2],[3,2],[4,3],[5,4],[7,5],[12,34],[11,31],[7,40],[9,36],[8,33],[1,5],[32,52],[35,51],[32,54],[34,50]])
obj=MinibatchKmeans(points,3,5)
print(obj.fit())


#obj2=MinibatchKmeans(points,3,5)
#print(obj2.minibatch(points,7))
