# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 13:36:00 2019

@author: drone911
"""
import numpy as np
from keras.datasets import mnist
import keras.backend as K
from keras.initializers import RandomNormal
from keras.layers import *
from keras.models import Model, Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from tqdm import tqdm

K.set_image_dim_ordering('tf')
class CatGAN():
    
    def __init__(self, img_size, num_channels, num_classes, latent_dim, generator_start_dim,G_LR, D_LR, BETA_1=0.5, BETA_2=0.999, verbose=False):
        self.img_size=img_size
        self.num_channels=num_channels
        self.img_shape=(img_size, img_size, num_channels)
        self.num_classes=num_classes
        self.latent_dim=latent_dim
        self.generator_start_dim=generator_start_dim
        self.generator=self._make_generator()
        self.discriminator=self._make_discriminator()
        
        self.generator.compile(optimizer=Adam(lr=G_LR, beta_1=BETA_1, beta_2=BETA_2), loss='binary_crossentropy')
        self.discriminator.compile(optimizer=Adam(lr=D_LR, beta_1=BETA_1, beta_2=BETA_2), loss=['binary_crossentropy',self._mutual_loss])
        
        latent_var=Input(shape=(self.latent_dim,))
        label=Input(shape=(1, ))
        gen_output=self.generator([latent_var, label])
        self.discriminator.trainable=False
        valid, output_label=self.discriminator(gen_output)
        self.catGAN=Model(inputs=[latent_var, label], outputs=[valid, output_label])
        self.catGAN.compile(optimizer=Adam(lr=G_LR, beta_1=BETA_1, beta_2=BETA_2), loss=['binary_crossentropy', self._mutual_loss])
        
        if verbose:
            print(self.discriminator.summary())
            print(self.catGAN.summary())
        
    def save_model(self, path_d, path_g):
        self.discriminator.save_weights(path_g)
        self.generator.save_weights(path_d)
        
    def load_model(self, path_d, path_g):
        self.generator.load_weights(path_g)
        self.discriminator.load_weights(path_d)
        
    def fit(self, train_x, train_y, epochs=100, batch_size=128):
        num_train_images=train_x.shape[0]
        num_batches=int(num_train_images/batch_size)
        hist_disc_avg, hist_gen_avg=[], []
        #train_y=to_categorical(train_y, num_classes=self.num_classes)
        
        for e in range(epochs):
            real_img_y=np.zeros((batch_size, 1))
            real_img_y[:]=0.9
            fake_img_y=np.zeros((batch_size, 1))
            hist_disc,hist_gen=[], []
            iterator=tqdm(range(num_batches))
            try:
                for i in iterator:
                    noise = np.random.normal(0,1.0,size=(batch_size, self.latent_dim))
                    
                    idx=np.random.randint(0, num_train_images, batch_size)
                    
                    real_batch_x=train_x[idx]
                    real_batch_y=train_y[idx]
                    fake_batch_y=np.random.randint(0, self.num_classes, batch_size).reshape(-1)
                    
                    
                    fake_batch_x=self.generator.predict([noise, fake_batch_y])
                    
                    train_disc_x=np.concatenate((real_batch_x, fake_batch_x), axis=0)
                    train_disc_y1=np.concatenate((real_img_y, fake_img_y), axis=0)
                    
                    train_disc_y2=np.concatenate((real_batch_y, fake_batch_y))
                    
                    train_disc_y2=to_categorical(train_disc_y2, num_classes=self.num_classes)
                    
                    self.discriminator.trainable=True
                    
                    hist_disc.append(self.discriminator.train_on_batch(train_disc_x, [train_disc_y1, train_disc_y2]))
                    
                    noise = np.random.normal(0,1.0,size=(2*batch_size, self.latent_dim))
                    sampled_labels=np.random.randint(0, self.num_classes, 2*batch_size).reshape(-1)
                    ones=np.ones((2*batch_size,1))
                    ones[:]=0.9
                    
                    sampled_labels_oh=to_categorical(sampled_labels, num_classes=self.num_classes)
                    
                    self.discriminator.trainable=False
                    hist_gen.append(self.catGAN.train_on_batch([noise, sampled_labels], [ones, sampled_labels_oh]))
                    
                hist_disc_avg.append(np.mean(hist_disc[:]))
                hist_gen_avg.append(np.mean(hist_gen[:]))
                print("-------------------------------------------------------")
                print("discriminator loss at epoch {}:{}".format(e, hist_disc_avg[-1]))
                print("generator loss at epoch {}:{}".format(e, hist_gen_avg[-1]))
                print("-------------------------------------------------------")
                self._plot_loss(hist_disc_avg, hist_gen_avg)
                self._sample_images(e)
                #if e % 10 == 0:
                    #self.save_model("models", e)
            except KeyboardInterrupt:
                print("Interrupted")
                iterator.close()
                break
                
            
    def predict(self, input_label):
        noise=np.random.normal(0,1.0,size=(self.latent_dim,1))
        return self.generator([noise, input_label])
    
    def _make_generator(self):
        init=RandomNormal(mean=0.0, stddev=0.02)
        '''
        model=Sequential()
        
        #input_shape
        model.add(Dense(np.prod(self.generator_start_dim), kernel_initializer=init, input_dim=self.latent_dim))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape(self.generator_start_dim))
        
        #8x8
        model.add(UpSampling2D(size=2))
        model.add(Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=init))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        
        #16x16
        model.add(UpSampling2D(size=2))
        model.add(Conv2D(32, kernel_size=5, strides=1, padding='valid', kernel_initializer=init))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        
        #28x28
        model.add(Conv2D(self.num_channels, kernel_size=3, strides=1, activation='tanh',padding='same', kernel_initializer=init))
        '''
        model = Sequential()

        model.add(Dense(1024, input_dim=self.latent_dim, activation='relu', init=init))
        model.add(Dense(128 * 7 * 7, activation='relu', init=init))
        model.add(Reshape((7, 7, 128)))

        # upsample to (..., 14, 14)
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(256, kernel_size=5,strides=1 ,padding='same', activation='relu', init=init))

        # upsample to (..., 28, 28)
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(128, kernel_size=5,strides=1, padding='same', activation='relu', init=init))

        # take a channel axis reduction
        model.add(Conv2D(self.num_channels, kernel_size=2,strides=1, padding='same', activation='tanh', init=init))

        noise=Input(shape=(self.latent_dim, ))
        label=Input(shape=(1, ))
        
        embedding_label=Embedding(input_dim=10, output_dim=self.latent_dim)(label)
        embedded_label=Reshape([self.latent_dim])(embedding_label)
        gen_input=Lambda(lambda x: x[0]*x[1])([noise, label])
        
        gen_output=model(gen_input)
        generator=Model(inputs=[noise, label], outputs=gen_output)
        return generator
    
    def _make_discriminator(self, ):
        init=RandomNormal(mean=0.0, stddev=0.02)
        '''model=Sequential()
        #28x28
        model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', kernel_initializer=init, input_shape=self.img_shape))
        model.add(GaussianNoise(stddev=0.02))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        
        #14x14
        model.add(Conv2D(64, kernel_size=3, strides=2, padding='same', kernel_initializer=init))
        model.add(GaussianNoise(stddev=0.02))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        #7x7
        input_img=Input(shape=(self.img_shape))
        model_output=model(input_img)
        '''
        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, padding='same', strides=2,input_shape=self.img_shape, init=init))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))

        model.add(Conv2D(64, kernel_size=3, padding='same', strides=1, init=init))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, kernel_size=3, padding='same', strides=2, init=init))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))

        model.add(Conv2D(256, kernel_size=3, padding='same', strides=2, init=init))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))

        model.add(Flatten())
        
        input_img=Input(shape=(self.img_shape))
        model_output=model(input_img)
        
        valid=Dense(1, activation='sigmoid', kernel_initializer=init)(model_output)
        
        output_label=Dense(self.num_classes, activation='softmax', kernel_initializer=init)(model_output)
        
        discriminator=Model(inputs=input_img, outputs=[valid,output_label])
        
        
        return discriminator
    
    def _plot_loss(self, hist_disc, hist_gen):
        plt.figure(figsize=(12,8))
        plt.plot(hist_disc, label='discriminator loss')
        plt.plot(hist_gen, label='generator loss')
        plt.legend()
        plt.savefig('loss_plot.png')
        plt.close()
    def _mutual_loss(self,c, c_given_x):
        eps = 1e-8
        conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
        entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

        return conditional_entropy + entropy
    
    def _sample_images(self, epoch):
        noise=np.random.uniform(-1.0,1.0,size=(self.num_classes, self.latent_dim))
        labels=np.arange(0, self.num_classes)
        generated_images=self.generator.predict([noise, labels])
        generated_images=(generated_images + 1.0)*127.5
        plt.figure(figsize = (15, 8))
        for i in range(generated_images.shape[0]):
            plt.subplot(5, 5, i + 1)
            plt.imshow(generated_images[i,:,:,0].astype(np.uint8), interpolation='nearest', cmap='gray')
            plt.xlabel(str(labels[i]))
        plt.suptitle("Epoch {}".format(epoch), x = 0.5, y = 1.0)
        plt.tight_layout()
        plt.savefig("generated_images\\numbers_at_epoch_{}.png".format(epoch))
        
if __name__=="__main__":
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    train_x=np.concatenate((train_x, test_x), axis=0)
    train_y=np.concatenate((train_y, test_y), axis=0)
    
    idx=np.arange(0, train_x.shape[0])
    np.random.shuffle(idx)
    train_x=train_x[idx]
    train_y=train_y[idx]
    train_x=(train_x/127.5)-1.0 
    train_x=train_x.reshape(-1,28,28,1)
    
    gan=CatGAN(img_size=28, num_channels=1, num_classes=10, latent_dim=100, generator_start_dim=(8,8,96),G_LR=0.0002, D_LR=0.0002, BETA_1=0.5, BETA_2=0.999,verbose=False)
    gan.fit(train_x, train_y, epochs=100, batch_size=128)
    
    