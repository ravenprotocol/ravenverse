import ravop.core as R
from ravop.utils import inform_server
from ravop.core import Graph, Tensor, Scalar, square_root,add, min
'''
                Using newly added Ravop Operations
'''

class Kmeans():
    def __init__(self,**kwargs):
        pass

    def initialize_centroids(self):
        cen=R.random(self.points, size=self.k)
        print(cen.output)
        while cen.status!='computed':
            pass
        cen=cen.reshape(shape= [ self.k,len(self.points.output[0])])
        inform_server()
        print(cen)
        return cen

    def closest_centroids(self):
        closest = R.argmin(square_root(  R.sub( self.points  ,self.centroids.expand_dims(axis=1) ).pow(Scalar(2)).sum(axis=2)))
        #np.array(self.points) - np.array(self.centroids.output)[:, np.newaxis]
        while closest.status != 'computed':
            pass
        inform_server()
        #print(closest.output)


        return closest

    def update_centroids(self):

        newcen=list()
        for i in range(self.k):
            mean = Tensor(self.points.output[self.label.output == i] ).mean(axis=0)
            print("1")
            while mean.status != 'computed':
                pass
            print(mean.output)
            inform_server()
            newcen.append(mean.output[0])

        print("======")
        inform_server()

        return Tensor(newcen)


    def fit(self,X,k,iter=50):
        self.points=Tensor(X)
        self.k=k
        self.iter=iter
        self.centroids=self.initialize_centroids()
        self.label=self.closest_centroids()
        for i in range(iter):
            print('iteration',i)
            self.centroids=self.update_centroids()
            self.label = self.closest_centroids()


        return self.label
        pass

obj = Kmeans()
#x=obj.fit( [[1,2],[3,2],[4,3],[5,4],[7,5],[12,34],[11,31],[7,40],[9,36],[8,33],[1,5],[32,52],[35,51],[32,54],[34,50]], 3,iter=20)
x=obj.fit( [[1,2,3],[3,2,4],[4,3,3],[5,4,123],[7,5,544],[12,34,444],[11,31,234],[7,40,412],[9,36,453],[8,33,433],[1,5,10001],[32,52,10003],[35,51,10031],[32,54,1003],[34,50,10031]], 3,iter=100)

print(obj.centroids.output)
print(x)
#print(obj.initialize_centroids().output)
#print(obj.closest_centroids() )



'''
# Using Ravop opeartions

class Kmeans():
    def __init__(self,X,k,max_iter=300):
        self.points=X
        self.k=k
        self.iter=max_iter

    def initialize_centroids(self):

        print("___",self.points)
        self.centroids=Tensor(random.sample(self.points,self.k))
        #self.centroids=Tensor(self.points)
        #inform_server()
        #return self.centroids

    def closest_centroid(self):
        closest = R.div(Scalar(1) ,square_root((Tensor( np.array(self.points)- np.array(self.centroids.output)[:, np.newaxis]   ).pow(Scalar(2))).sum(axis=2)))
        while closest.status != 'computed':
            pass
        inform_server()
        #print(closest)


        self.label=R.argmax(closest)
        while self.label.status!= 'computed':
            pass
        inform_server()
        return self.label

    def update_centroids(self):
        newcen=[]

        #print("label output:",self.label.output)
        #print("array of self.points", (self.points) )
        #for x in range(len(self.label.output)):
        #    print(self.label.output[x])


        for i in range(self.k):
            newcen.append(np.mean([self.points[x] for x in range(len(self.label.output)) if self.label.output[x] == i],axis=0) )#if self.label.output[x] == Scalar(2)])
        #print(newcen[0],"\n",newcen[1],"\n",newcen[2])
        return Tensor(newcen)

    def fit(self):
        self.initialize_centroids()
        self.label=self.closest_centroid()
        for i in range(self.iter):
            print("iteration ",i)
            self.centroids= self.update_centroids()
            self.label= self.closest_centroid()
            while self.label.status!='computed':
                pass
            inform_server()

        return self.labe
        pass


obj = Kmeans( [[1,2],[3,2],[4,3],[5,4],[7,5],[12,34],[11,31],[7,40],[9,36],[8,33],[1,5],[32,52],[35,51],[32,54],[34,50]], 3)

print("Labels after clustering :")
print(obj.fit().output)


#print(obj.fit())
#inform_server()
'''


'''
# Using numpy 
class K_means():
    def __init__(self,points,k,max_iter=300):
        self.points=points
        self.k=k
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
        return np.argmin(distances, axis=0)

    def update_centroids(self):
        newcen=[]
        for k in range(self.k):
            p=points[self.label==k].mean(axis=0)
            newcen.append(p)

        return np.array(newcen)

    def fit(self):
        #update
        for i in range(self.iter):
            self.centroids = self.update_centroids()
            self.label = self.closest_centroid(points, self.centroids)
            pass
        print(self.centroids)

        return self.label


    def plot_scatter(self,label):
        fig, axs = plt.subplots(2)

        axs[0].scatter(points[:, 0], self.points[:, 1], c=label )
        axs[1].scatter(points[:, 0], self.points[:, 1], c=self.temp)
        axs[0].legend()
        plt.show()

    def predict(self,point):
        return self.closest_centroid(np.array(point), self.centroids)
        pass
'''