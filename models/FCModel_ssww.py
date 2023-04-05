from tensorflow             import keras
import tensorflow as tf
from keras.layers           import Input, Dense, Dropout, Lambda, Concatenate, Conv1D, BatchNormalization
from tensorflow.python.ops import math_ops
from models.Model_ssww import Model_Functional as Model

class FCModel_Functional(Model):
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

    self.input_layer = Input(shape=(len(self.FEATURES),),name="input_layer")
    self.normalization = keras.layers.Normalization(axis=-1,
      mean      = [FCModel_Functional.MEAN(self.dframe[k]) for k in self.FEATURES],
      variance  = [FCModel_Functional.VAR (self.dframe[k]) for k in self.FEATURES])(self.input_layer)
    self.normalization.trainable = False
    self.layer1 = Dense(self.neurons,**self.DENSE_SETUP,name="layer1")(self.normalization)    
    self.layer2 = Dense(self.neurons,**self.DENSE_SETUP,name="layer2")(self.layer1)
    self.layer3 = Dense(self.neurons,**self.DENSE_SETUP,name="layer3")(self.layer2)
    self.layer4 = Dense(self.neurons,**self.DENSE_SETUP,name="layer4")(self.layer3)
    self.layer5 = Dense(self.neurons,**self.DENSE_SETUP,name="layer5")(self.layer4)
    self.layer6 = Dense(self.neurons,**self.DENSE_SETUP,name="layer6")(self.layer5)
    self.layer7 = Dense(self.neurons,**self.DENSE_SETUP,name="layer7")(self.layer6)
    self.layer8 = Dense(self.neurons,**self.DENSE_SETUP,name="layer8")(self.layer7)     
    self.neutrini_layer = Dense(self.n_targets, **self.LAST_SETUP,name="neutrini_layer")(self.layer8)    
    self.model = tf.keras.models.Model(self.input_layer, self.neutrini_layer, name=self.name)        
    print(self.model.summary())
    
    self.model.compile(**self.COMPILE_SETUP,loss=tf.keras.losses.MAE,run_eagerly=False)
    #self.model.compile(**self.COMPILE_SETUP,run_eagerly=False)
