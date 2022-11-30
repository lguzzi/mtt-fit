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
  
  def dummyLoss(self):    
    output = self.model.get_layer("output_layer").output 
    inputs= self.model.get_layer("normalization").output
    #higgs_px = inputs.to_numpy()[0]+ inputs.to_numpy()[4]+ output.to_numpy()[0] + output.to_numpy()[4]
    tau1 = inputs[:,0:3]
    nu1 = output[:,0:3]
    tau2 = inputs[:,4:7]
    nu2 = output[:,4:7]
    return tf.reduce_mean(nu1)
    #higgs_py = inputs[:][1] + inputs[:][5]+ output[:][1] + output[:][5]
    #higgs_pz = inputs[:][2] + inputs[:][6]+ output[:][2] + output[:][6]   
    #higgs_e= inputs[:][3] + inputs[:][7] + output[:][3] + output[:][7]
    #mass2 = tf.square(higgs_e) - tf.square(higgs_px) - tf.square(higgs_py) - tf.square(higgs_pz)
    #red_mass2 = tf.reduce_mean(mass2)
    #print("reduced squared mass ", red_mass2) 
    #return red_mass2

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
