""" 
trains both the gan and dgan
"""
import numpy as np
import tensorflow as tf 
import math

from config import *
from nn import *

#{x,y,r}[]
circles = np.array([
    [0.25, 0.25, 0.2],
    [0.8, 0.12, 0.075],
    [0.7, 0.75, 0.15]
])

#circles = np.array([[0.05,0.75,0.05], [0.85, 0.85,0.05]])

def training_data(N):
    #pi is factored out
    normalized_areas = circles[:, 2] ** 2 
    normalized_areas /= np.sum(normalized_areas); 

    #square root for uniform distribution
    rmul = tf.math.sqrt(tf.random.uniform([N], 0, 1));
    theta = tf.random.uniform([N], 0, 2 * math.pi); 
    index = tf.random.categorical(tf.math.log([normalized_areas]), N)[0]

    c = tf.math.cos(theta).numpy();
    s = tf.math.sin(theta).numpy();

    circ = circles[index.numpy()]
    x = circ[:, 0]  
    y = circ[:, 1]  
    r = circ[:, 2] * rmul

    pos = np.array([x + c * r, y + s * r]).T
    return tf.cast(tf.constant(pos), tf.float32);

def save_models():
    pass

if (__name__ == '__main__'):
    print("Starting training...");
    #train gan using helper function
    for _ in dgan_train(training_data(DATA_ELEMENTS * DGAN_COMPARE_ENTRIES)):
        pass
    #train dgan using helper function
    for _ in gan_train(training_data(DATA_ELEMENTS)):
        pass

    #use evaluate and save output
    save_models();
