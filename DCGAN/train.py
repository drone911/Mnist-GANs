# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 20:11:22 2019

@author: drone911
"""

from helper import *
from models import *
import numpy as np
from keras.datasets import mnist
from tqdm import tqdm
import warnings

def train(train_images, generator, discriminator, gan, num_classes=120, random_dim=128, epochs=100, batch_size=128):
    num_train_images=train_images.shape[0]
    num_batches=int(num_train_images/batch_size)
    hist_disc_avg, hist_gen_avg=[], []
    
    for e in range(epochs):
        fake_img_y=np.zeros((batch_size, 1))
        fake_img_y[:]=0
        real_img_y=np.zeros((batch_size, 1))
        real_img_y[:]=0.9
        gan_y=np.ones((batch_size, 1))
        hist_disc, hist_gen=[], []
        iterator=tqdm(range(num_batches))
        try:
            for i in iterator:
                sampled_noise = generate_inputs(random_dim, batch_size)
                
                real_img_x=train_images[np.random.randint(0,train_images.shape[0],size=batch_size)]
                fake_img_x=generator.predict(sampled_noise)
                
                train_disc_x=np.concatenate((real_img_x, fake_img_x), axis=0)
                train_disc_y=np.concatenate((real_img_y, fake_img_y), axis=0)
                
                discriminator.trainable=True
                hist_disc.append(discriminator.train_on_batch(train_disc_x, train_disc_y))
                
                noise=generate_inputs(random_dim, batch_size)
                
                discriminator.trainable=False
                hist_gen.append(gan.train_on_batch(noise, gan_y))
                
            hist_disc_avg.append(np.mean(hist_disc[0:num_batches]))
            hist_gen_avg.append(np.mean(hist_gen[0:num_batches]))
            print("-------------------------------------------------------")
            print("discriminator loss at epoch {}:{}".format(e, hist_disc_avg[-1]))
            print("generator loss at epoch {}:{}".format(e, hist_gen_avg[-1]))
            print("-------------------------------------------------------")
            
            plot_generated_images(e, generator, random_dim=random_dim)
            plot_loss(hist_disc, hist_gen)
            if e % 10 == 0:
                discriminator.save_weights("models\\disc_v1_epoch_{}.h5".format(e))
                generator.save_weights("models\\gen_v1_epoch_{}.h5".format(e))
        except KeyboardInterrupt:
            iterator.close()
            print("Interrupted")
            break
if __name__=="__main__":    
    
    warnings.filterwarnings("ignore")
    (train_images, train_labels), (test_images, test_labels)=mnist.load_data()
    random_dim=100
    batch_size=128
    lr=0.0002
    beta_1=0.5
    train_images=np.concatenate((train_images, test_images), axis=0)
    train_images=train_images.reshape(-1,28,28,1)
    train_images=(train_images.astype(np.float32) - 127.5) / 127.5

    generator=get_gen_nn(random_dim=random_dim, lr=lr, beta_1=beta_1,verbose=False)
    discriminator=get_disc_nn(lr=lr, beta_1=beta_1,verbose=False)
    
    gan=create_gan(discriminator, generator, random_dim=random_dim, lr=lr, beta_1=beta_1,verbose=False)
    train(train_images, generator, discriminator, gan, random_dim=random_dim, epochs=50, batch_size=128)    
