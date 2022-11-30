from __future__ import print_function

import pickle as pkl

import numpy as np
import ravop as R

#image preprocessing :

from os import listdir
from os.path import isfile, join
from keras_preprocessing import image
from keras.applications.vgg16 import preprocess_input
from numpy import moveaxis



from ravdl.v2 import NeuralNetwork
from ravdl.v2.layers import Activation, Dense, BatchNormalization, Dropout, Conv2D, Flatten, MaxPooling2D
from ravdl.v2.optimizers import Adam, RMSprop
from ravdl.v2.loss_functions import CrossEntropy

from sklearn import datasets
from sklearn.model_selection import train_test_split
imagenette_map = { 
    "n01440764" : "tench",
    "n02102040" : "springer",
    "n02979186" : "casette_player",
    "n03000684" : "chain_saw",
    "n03028079" : "church", 
    "n03394916" : "French_horn",
    "n03417042" : "garbage_truck",
    "n03425413" : "gas_pump",
    "n03445777" : "golf_ball",
    "n03888257" : "parachute"
}
label_y={
    "tench":0,
    "springer":1,
    "casette_player":2,
    "chain_saw":3,
    "church":4, 
    "French_horn":5,
    "garbage_truck":6,
    "gas_pump":7,
    "golf_ball":8,
    "parachute":9

}

def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def get_dataset(path):
    v=0
    
    img_arr=[]
    y_label=[]
    for _ in imagenette_map.keys():
        onlyfiles = [f for f in listdir(path+_) ]
        # print(len(onlyfiles))
        lab_len=len(onlyfiles)
        y=label_y[ imagenette_map[_]]
        for f in onlyfiles:
            path_img= path+_+"/"+str(f)
            img = image.load_img(path_img, target_size=(224, 224))
            img = image.img_to_array(img)
            # img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            img= moveaxis(img, 2, 0)

            img_arr.append(img)
            # print("img_shape=",np.shape(img))
            y_label.append(y)
    return img_arr, to_categorical(np.array(y_label),n_col=1000)



# creating the vgg 16 model:

def create_model():
    optimizer = Adam()
    model = NeuralNetwork(optimizer=optimizer, loss='CrossEntropy')
    model.add(Conv2D(input_shape=(3,224,224),n_filters=64,filter_shape=(3,3),padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters=64,filter_shape=(3,3),padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_shape=(2,2),stride=2))

    model.add(Conv2D(n_filters=128, filter_shape=(3,3), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters=128, filter_shape=(3,3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_shape=(2,2),stride=2))

    model.add(Conv2D(n_filters=256, filter_shape=(3,3), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters=256, filter_shape=(3,3), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters=256, filter_shape=(3,3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_shape=(2,2),stride=2))

    model.add(Conv2D(n_filters=512, filter_shape=(3,3), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters=512, filter_shape=(3,3), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters=512, filter_shape=(3,3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_shape=(2,2),stride=2))

    model.add(Conv2D(n_filters=512, filter_shape=(3,3), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters=512, filter_shape=(3,3), padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters=512, filter_shape=(3,3), padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_shape=(2,2),stride=2))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dense(1000))
    model.add(Activation('softmax'))
    model.summary()
    return model

def train(model, X_train, y_train,batch_size=256, n_epochs=10):
    model.fit(X_train, y_train, n_epochs=n_epochs, batch_size=batch_size, persist_weights=False) #v2
    # model.fit(X_train, y_train, n_epochs=n_epochs, batch_size=256, save_model=True) #v1
    # pkl.dump(model, open("cnn_model.pkl", "wb"))
    return model

def test(model, X_test, y_test):
    print('\nTesting...')
    loss, acc = model.test_on_batch(R.t(X_test), R.t(y_test))
    loss.persist_op(name='cnn_test_loss')
    acc.persist_op(name='cnn_test_acc')

def compile():
    R.activate()

def execute():
    R.execute()
    R.track_progress()



