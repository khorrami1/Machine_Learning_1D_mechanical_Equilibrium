# Importing modules
import os
from matplotlib.pyplot import axis
import numpy as np
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

# n: number of discretization
n = 100
x = np.linspace(0,1,n)

# number of samples
num_samples = 200

# C: Young's modulus
C = np.random.random((num_samples,n))

# Ebar: the mean value of the strain
Ebar = np.random.random((num_samples,1))/2 # strain in the range (0,0.5)

inputs = np.concatenate((C,Ebar),axis=1)


class Custom_Loss(keras.losses.Loss):
    def __init__(self, inputs):
        super().__init__()
        n = inputs.shape[1]-1
        self.C = tf.cast(inputs[:,0:n],'float32')
        self.Ebar = tf.cast(inputs[:,n],'float32')

    def call(self, y_true, y_pred):
        stress = tf.multiply(self.C,y_pred)
        # divergence of stress is zero.
        grad_stress = tf.Variable(np.gradient(stress,axis=1)) # Error: Cannot convert a symbolic Tensor (Custom_Loss/Mul:0) to a numpy array
        MSE_PDE = tf.math.reduce_mean(grad_stress,axis=1) # must be corrected
        MSE_BC = tf.math.reduce_mean(y_pred-self.Ebar) # must be corrected
        return MSE_PDE + MSE_BC

# model = Sequential(
#     Dense(101,activation='linear'),
#     Dense(50,activation='tanh'),
#     Dense(100,activation='linear'),
# )

def myNet():
    layer0 = keras.layers.Input((n+1))
    layer1 = Dense(101,activation='linear')(layer0)
    layer2 = Dense(20,activation='relu')(layer1)
    outputs =  Dense(100,activation='linear')(layer2)
    model = keras.Model(inputs=layer0, outputs=outputs)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # 4.1 -- The loss is chosen to be mean absolute error
    #model.compile(optimizer=optimizer, loss=['mean_absolute_error'], metrics=['mean_absolute_error'])
    # 4.2 -- Just print the model's summary for information
    #model.summary()
    return model

model = myNet()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

model.compile(optimizer=optimizer, loss=Custom_Loss(inputs), metrics=['mean_absolute_error'])
model.fit(inputs, np.zeros((num_samples,n)), batch_size=32, epochs=100)
