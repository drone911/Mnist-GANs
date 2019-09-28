# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 17:17:13 2019

@author: drone911
"""
from keras.layers import *
from keras.initializers import RandomNormal
from keras import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import Model
from keras import Input



def get_disc_nn(input_shape=(28,28,1), lr=0.0001, beta_1=0.5, weights_path="", verbose=False):

    model= Sequential()
    model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', input_shape=input_shape, kernel_initializer=RandomNormal(stddev=0.02)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr, beta_1=beta_1))

    if weights_path:
        try:
            model.load_weights(weights_path)
            print("weights loaded")
        except:
            print("weights were not loaded")
    if verbose:
        print(model.summary())
    return model


def get_gen_nn(start_dim=128*7*7, random_dim=128, lr=0.0002, beta_1=0.5, weights_path="", verbose=False):
    
    model = Sequential()
    model.add(Dense(128*7*7, input_dim=random_dim, kernel_initializer=RandomNormal(stddev=0.02)))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7, 7, 128)))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=5, padding='same'))
    model.add(LeakyReLU(0.2))

    model.add(UpSampling2D())
    model.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))
    
    if weights_path:
        try:
            model.load_weights(weights_path)
            print("weights loaded")
        except:
            print("weights were not loaded")
    if verbose:
        print(model.summary())
    return model


def create_gan(disc_nn, gen_nn ,random_dim=128 , lr=0.0002, beta_1=0.5, verbose=False):
    disc_nn.trainable=False
    
    gen_input=Input(shape=(random_dim, ))
    
    gen_output= gen_nn(gen_input)
    disc_output=disc_nn(gen_output)
    
    model=Model(inputs=gen_input, outputs= disc_output)
    model.compile(optimizer=Adam(lr=lr, beta_1= beta_1), loss='binary_crossentropy')
    
    if verbose:
        print(model.summary())
    return model

