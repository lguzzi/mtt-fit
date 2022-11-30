from tensorflow             import keras
import tensorflow as tf
from keras.layers           import Input, Dense, Dropout, Lambda, Concatenate
from tensorflow.python.ops import math_ops
from models.Model_Functional import Model_Functional as Model

class FCModel_Functional(Model):
  MEAN = lambda x: x.mean() if x.dtype!='int16' else 0
  VAR  = lambda x: x.var()  if x.dtype!='int16' else 1
  def __init__(self, dropout, neurons, n_targets=1, **kwargs):
    super().__init__(**kwargs)
    self.dropout = dropout
    self.neurons = neurons
    self.n_targets = n_targets
  
  def minv2(tensor):
    taus = tensor[0]
    neutrinos = tensor[1]
    higgs_px = taus[:,0] + taus[:,4] + neutrinos[:,0]+neutrinos[:,4]
    higgs_py = taus[:,1] + taus[:,5] + neutrinos[:,1]+neutrinos[:,5]
    higgs_pz = taus[:,2] + taus[:,6] + neutrinos[:,2]+neutrinos[:,6]
    higgs_e = taus[:,3] + taus[:,7] + neutrinos[:,3]+neutrinos[:,7]
    minv2 = (higgs_e**2)-(higgs_px**2)-(higgs_py**2)-(higgs_pz**2) 
    #print("massa ",minv2)   
    return tf.expand_dims(minv2, axis=-1)

  def customMAE(y_true,y_pred):
    #print("mass ",y_pred[:,8],y_true[:,8])    
    #return tf.keras.backend.mean(math_ops.squared_difference(y_pred[:,:8],y_true[:,:8]),axis=-1)
    deltaNeutrini = tf.keras.backend.mean(tf.math.abs(y_pred[:,:8]-y_true[:,:8]),axis=-1)
    deltaMass2 = tf.keras.backend.mean(tf.math.abs(y_pred[:,8]-y_true[:,8]),axis=-1)
    alpha = 0.01
    return deltaNeutrini + alpha*deltaMass2 




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
    self.output_layer = Dense(self.n_targets, **self.LAST_SETUP,name="output_layer")(self.layer8)    
    self.taus = self.input_layer[:,:8]          
    self.minv2_ = Lambda(FCModel_Functional.minv2, name="layer_minv2")([self.taus,self.output_layer])
    self.mergedOutput = Concatenate(axis=-1)([self.output_layer, self.minv2_])
    self.model.add_loss(FCModel_Functional.customMAE)
    self.model = tf.keras.models.Model(self.input_layer, self.mergedOutput, name=self.name)    
    
    print(self.model.summary())
    
    #self.model.compile(**self.COMPILE_SETUP,loss = FCModel_Functional.customMAE,run_eagerly=False)
    sself.model.compile(**self.COMPILE_SETUP,run_eagerly=False)
