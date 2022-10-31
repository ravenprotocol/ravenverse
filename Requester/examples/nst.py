# from __future__ import print_function

import pickle as pkl

import numpy as np
import ravop as R


from ravdl.v2 import NeuralNetwork
from ravdl.v2.layers import Activation, Dense, BatchNormalization, Dropout, Conv2D, Flatten, MaxPooling2D
from ravdl.v2.optimizers import Adam, RMSprop
from ravdl.v2.loss_functions import CrossEntropy,SquareLoss

from sklearn import datasets
from sklearn.model_selection import train_test_split


def to_categorical(x, n_col=None):
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def get_dataset():

    from keras_preprocessing import image
    from keras.applications.vgg16 import preprocess_input
    img_path="/Users/pranitkandarpa/Desktop/style_image.jpeg"
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    print(img)

    print(np.shape(img))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    from numpy import moveaxis
    print(np.shape(img))
    img= moveaxis(img, 3, 1)
    # img = img.reshape((-1, 224, 224,3))
    print("=================\n\n\n")
    print(img,np.shape(img))
    return img, img ,[[1]] ,[[1]]
    pass



def create_model():#n_hidden, n_features):
    optimizer = Adam()
    model = NeuralNetwork(optimizer=optimizer, loss='SquareLoss')





    model.add(Conv2D(input_shape=(3,224,224),n_filters=64,filter_shape=(3,3),padding="same"))#, activation="relu"))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters=64,filter_shape=(3,3),padding="same"))#, activation="relu"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_shape=(2,2),stride=2))

    model.add(Conv2D(n_filters=128, filter_shape=(3,3), padding="same"))#, activation="relu"))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters=128, filter_shape=(3,3), padding="same"))#, activation="relu"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_shape=(2,2),stride=2))

    model.add(Conv2D(n_filters=256, filter_shape=(3,3), padding="same"))#, activation="relu"))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters=256, filter_shape=(3,3), padding="same"))#, activation="relu"))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters=256, filter_shape=(3,3), padding="same"))#, activation="relu"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_shape=(2,2),stride=2))

    model.add(Conv2D(n_filters=512, filter_shape=(3,3), padding="same"))#, activation="relu"))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters=512, filter_shape=(3,3), padding="same"))#, activation="relu"))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters=512, filter_shape=(3,3), padding="same"))#, activation="relu"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_shape=(2,2),stride=2))

    model.add(Conv2D(n_filters=512, filter_shape=(3,3), padding="same"))#, activation="relu"))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters=512, filter_shape=(3,3), padding="same"))#, activation="relu"))
    model.add(Activation('relu'))
    model.add(Conv2D(n_filters=512, filter_shape=(3,3), padding="same"))#, activation="relu"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_shape=(2,2),stride=2))

    model.add(Flatten())
    model.add(Dense(4096))#,activation="relu"))
    model.add(Activation('relu'))
    model.add(Dense(4096))#,activation="relu"))
    model.add(Activation('relu'))
    model.add(Dense(1000))#, activation="softmax"))
    model.add(Activation('softmax'))
    model.summary()

    return model




def train(model, X_train, y_train, n_epochs=5):
    model.fit(X_train, y_train, n_epochs=n_epochs, batch_size=1, persist_weights=True) #v2
    # model.fit(X_train, y_train, n_epochs=n_epochs, batch_size=256, save_model=True) #v1
    # pkl.dump(model, open("cnn_model.pkl", "wb"))
    return model

def forward_p(model,X_train):
    out=model._forward_pass(R.t(X_train),training=False,return_all_layer_output=True)
    return out


def test(model, X_test, y_test):
    print('\nTesting...')
    loss, acc = model.test_on_batch(R.t(X_test), R.t(y_test))
    # loss.persist_op(name='cnn_test_loss')
    # acc.persist_op(name='cnn_test_acc')

def compile():
    R.activate()


def execute():
    R.execute()
    R.track_progress()

def get_score():
    conv,act=[],[]
    for _ in range(3):    
        conv.append( R.fetch_persisting_op("val_{}_batch_{}".format('conv1',_)))
        print(np.shape( R.fetch_persisting_op("val_{}_batch_{}".format('conv1',_))))
        act .append(R.fetch_persisting_op("val_{}_batch_{}".format('activ_10_',_)))
        print(np.shape(R.fetch_persisting_op("val_{}_batch_{}".format('activ_10_',_))))
    # print(conv,act)
    # print( np.shape(conv[0]['result']), np.shape(act[0]['result']),np.shape(conv[1]['result']), np.shape(act[1]['result']),np.shape(conv[2]['result']), np.shape(act[2]['result']))
    # print("____________________________________________________________________")
    # print(conv[0].keys())