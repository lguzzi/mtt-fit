from tensorflow             import keras
import tensorflow as tf
from keras.layers           import Input, Dense, Dropout, Lambda, Concatenate, Conv1D, BatchNormalization
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

  def customLoss(alpha=1E-6,beta=10.,gamma=1):
    alpha_ = 1E-6
    beta_ = 10.
    gamma_ = gamma
    def customMAE(y_true,y_pred):
      #print("mass ",y_pred[:,8],y_true[:,8])    
      #return tf.keras.backend.mean(math_ops.squared_difference(y_pred[:,:8],y_true[:,:8]),axis=-1)
      deltaNeutrini = tf.keras.backend.mean(tf.math.abs(y_pred[:,:8]-y_true[:,:8]),axis=-1)      
      #huber_loss_neutrini = tf.keras.losses.Huber(delta=10)
      #deltaNeutrini = huber_loss_neutrini(y_true[:,:8],y_pred[:,:8])
      #deltaMass2 = tf.keras.backend.mean(tf.math.abs(y_pred[:,8]-y_true[:,8]),axis=-1)
      huber_loss_minv2 = tf.keras.losses.Huber(delta=400) # --> 20
      deltaMass2=huber_loss_minv2(y_true[:,8],y_pred[:,8]) # --> 1,y_pred[:,8]/y_true[:,8]
      #deltaMass2=huber_loss_minv2(1,y_pred[:,8]/y_true[:,8]) # --> 1,y_pred[:,8]/y_true[:,8]
      crossEntropy = tf.keras.losses.CategoricalCrossentropy()(y_true[:,9:],y_pred[:,9:])
      return gamma_*deltaNeutrini + alpha_*deltaMass2 + beta_*crossEntropy
    return customMAE




  def compile(self,part=None,alpha=1E-5,beta=10.,gamma=1.):
    self.train_part1 = True
    self.train_part2 = True
    if part ==1:
      self.train_part2 = False
    if part ==2:
      self.train_part1 = False

    self.DENSE_SETUP   = self.SETUP['DENSE'  ]
    self.LAST_SETUP    = self.SETUP['LAST'   ]
    self.COMPILE_SETUP = self.SETUP['COMPILE']

    self.input_layer = Input(shape=(len(self.FEATURES),),name="input_layer")
    self.normalization = keras.layers.Normalization(axis=-1,
      mean      = [FCModel_Functional.MEAN(self.dframe[k]) for k in self.FEATURES],
      variance  = [FCModel_Functional.VAR (self.dframe[k]) for k in self.FEATURES])(self.input_layer)
    self.normalization.trainable = False
    self.layer1 = Dense(self.neurons,**self.DENSE_SETUP,name="layer1",trainable=self.train_part1)(self.normalization)
    #self.layer1 = tf.expand_dims(self.layer1,axis=1)
    #self.layerConv = Conv1D(10,1,activation='relu',name="layerConv")(self.layer1) 
    self.layer2 = Dense(self.neurons,**self.DENSE_SETUP,name="layer2",trainable=self.train_part1)(self.layer1)
    self.layer3 = Dense(self.neurons,**self.DENSE_SETUP,name="layer3",trainable=self.train_part1)(self.layer2)
    self.layer4 = Dense(self.neurons,**self.DENSE_SETUP,name="layer4",trainable=self.train_part1)(self.layer3)
    self.layer5 = Dense(self.neurons,**self.DENSE_SETUP,name="layer5",trainable=self.train_part1)(self.layer4)
    self.layer6 = Dense(self.neurons,**self.DENSE_SETUP,name="layer6",trainable=self.train_part1)(self.layer5)
    self.layer7 = Dense(self.neurons,**self.DENSE_SETUP,name="layer7",trainable=self.train_part1)(self.layer6)
    self.layer8 = Dense(self.neurons,**self.DENSE_SETUP,name="layer8",trainable=self.train_part1)(self.layer7)  
    #self.layer9 = Dense(self.neurons,**self.DENSE_SETUP,name="layer9",trainable=self.train_part1)(self.layer8)  
    #self.layer10 = Dense(self.neurons,**self.DENSE_SETUP,name="layer10",trainable=self.train_part1)(self.layer9)  
    #self.layer11 = Dense(self.neurons,**self.DENSE_SETUP,name="layer11",trainable=self.train_part1)(self.layer10)  
    #self.layer12 = Dense(self.neurons,**self.DENSE_SETUP,name="layer12",trainable=self.train_part1)(self.layer11)  
    #self.layer8=tf.squeeze(self.layer8,axis=1)    
    self.neutrini_layer = Dense(self.n_targets, **self.LAST_SETUP,name="neutrini_layer",trainable=self.train_part1)(self.layer8)    
    self.taus = self.input_layer[:,:8]          
    self.minv2_ = Lambda(FCModel_Functional.minv2, name="layer_minv2")([self.taus,self.neutrini_layer])
    self.mergingForClassification = Concatenate(axis=-1)([self.input_layer,self.neutrini_layer, self.minv2_])
    #self.mergingForClassification = self.input_layer
    self.layer21 = Dense(self.neurons,**self.DENSE_SETUP,name="layer21",trainable=self.train_part2)(self.mergingForClassification)
    self.layer31 = Dense(self.neurons,**self.DENSE_SETUP,name="layer31",trainable=self.train_part2)(self.layer21)
    self.layer41 = Dense(self.neurons,**self.DENSE_SETUP,name="layer41",trainable=self.train_part2)(self.layer31)
    self.layer51 = Dense(self.neurons,**self.DENSE_SETUP,name="layer51",trainable=self.train_part2)(self.layer41)
    self.layer61 = Dense(self.neurons,**self.DENSE_SETUP,name="layer61",trainable=self.train_part2)(self.layer51)
    self.layer71 = Dense(self.neurons,**self.DENSE_SETUP,name="layer71",trainable=self.train_part2)(self.layer61)
    self.layer81 = Dense(self.neurons,**self.DENSE_SETUP,name="layer81",trainable=self.train_part2)(self.layer71)
    self.classifier = Dense(3,activation=tf.keras.activations.softmax,name="classifier",trainable=self.train_part2)(self.layer81) 
    self.mergedOutput = Concatenate(axis=-1)([self.neutrini_layer, self.minv2_,self.classifier])    
    self.model = tf.keras.models.Model(self.input_layer, self.mergedOutput, name=self.name)        
    print(self.model.summary())
    
    self.model.compile(**self.COMPILE_SETUP,loss=FCModel_Functional.customLoss(alpha,beta,gamma),run_eagerly=False)
    #self.model.compile(**self.COMPILE_SETUP,run_eagerly=False)
