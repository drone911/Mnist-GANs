# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 15:23:32 2019

@author: drone911
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_inputs(random_dim, batch_size=128):
    sampled_noise = np.random.normal(0, 1, (batch_size, random_dim)) 
    return sampled_noise


def plot_generated_images(epoch, generator,examples = 25, dim = (5, 5), random_dim=128):
    noise=generate_inputs(random_dim, examples)
    generated_images=generator.predict(noise)
        
    plt.figure(figsize = (12, 8))
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i,:,:,0], interpolation='nearest', cmap='gray')
    plt.suptitle("Epoch {}".format(epoch), x = 0.5, y = 1.0)
    plt.tight_layout()
    plt.savefig("images\\numbers_at_epoch_{}.png".format(epoch))
    
def plot_loss(d, g):
    plt.figure(figsize = (18, 12))
    plt.plot(d, label = 'Discriminator Loss')
    plt.plot(g, label = 'Generator Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()
