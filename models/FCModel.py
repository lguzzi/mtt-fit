from tensorflow             import keras
import tensorflow as tf
from keras                  import Sequential
from keras.layers           import InputLayer, Dense, Dropout, Lambda

from models.Model import Model

class FCModel(Model):
  MEAN = lambda x: x.mean() if x.dtype!='int16' else 0
  VAR  = lambda x: x.var()  if x.dtype!='int16' else 1
  def __init__(self, dropout, neurons, n_targets=1, **kwargs):
    super().__init__(**kwargs)
    self.dropout = dropout
    self.neurons = neurons
    self.n_targets = n_targets
  
  def compile(self):
    self.DENSE_SETUP   = self.SETUP['DENSE'  ]
    self.LAST_SETUP    = self.SETUP['LAST'   ]
    self.COMPILE_SETUP = self.SETUP['COMPILE']

    self.model = Sequential(InputLayer(input_shape=len(self.FEATURES)), name=self.name)
    self.model.add(keras.layers.Normalization(axis=-1,
      mean      = [FCModel.MEAN(self.dframe[k]) for k in self.FEATURES],
      variance  = [FCModel.VAR (self.dframe[k]) for k in self.FEATURES])
    ) ; self.model.layers[-1].trainable = False
  

    for n in self.neurons:
      self.model.add(Dense(n, **self.DENSE_SETUP))
      if self.dropout is not None:
        #self.model.add(Dropout(self.dropout))
        self.model.add(Dropout(0.05))
    self.model.add(Dense(self.n_targets, **self.LAST_SETUP,name="output_layer"))
   
    print(self.model.summary())
    #self.model.add_loss(self.dummyLoss)
    self.model.compile(**self.COMPILE_SETUP)
