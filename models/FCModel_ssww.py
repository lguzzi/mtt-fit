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

  def minv2(tensor):
    l1 = tensor[0]
    v1 = tensor[1]   
    #v1e = tf.math.sqrt(v1[:,0]*v1[:,0] +v1[:,1]*v1[:,1] + v1[:,2]*v1[:,2] )    
    v1e = v1[:,0]
    w1_e = l1[:,0] + v1e
    w1_px = l1[:,1] + v1[:,1]
    w1_py = l1[:,2] + v1[:,2]
    w1_pz = l1[:,3] + v1[:,3]  
    
    #minv2 = tf.math.abs((w1_e**2)-(w1_px**2)-(w1_py**2)-(w1_pz**2))
    minv2 = tf.math.abs((w1_e**2)-(w1_px**2)-(w1_py**2)-(w1_pz**2))
    return tf.expand_dims(minv2,axis = -1)
    
    
    
  
  def customLoss(alpha):
    alpha_ = alpha
    def customMAE(y_true,y_pred):
      #print("mass ",y_pred[:,:3],y_true[:,:3])    
      #return tf.keras.backend.mean(math_ops.squared_difference(y_pred[:,:8],y_true[:,:8]),axis=-1)
      deltaNeutrini = tf.keras.backend.mean(tf.math.abs(y_pred[:,:8]-y_true[:,:8]),axis=-1)   
      huber_loss_minv2 = tf.keras.losses.Huber(delta=100) # --> 20
      deltaMass2_1=huber_loss_minv2(y_true[:,8],y_pred[:,8])         
      deltaMass2_2=huber_loss_minv2(y_true[:,9],y_pred[:,9])                           
      deltaMass2 = deltaMass2_1+deltaMass2_2
      return deltaNeutrini + alpha_*deltaMass2      
    return customMAE
  
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
    self.neutrini_layer = Dense(8, **self.LAST_SETUP,name="neutrini_layer")(self.layer8)
    self.l1 = self.input_layer[:,:4]
    self.l2 = self.input_layer[:,4:8]  
    self.v1 = self.neutrini_layer[:,:4]
    self.v2 = self.neutrini_layer[:,4:8]
    self.minv2_1 = Lambda(FCModel_Functional.minv2, name="layer_minv2_1")([self.l1,self.v1]) 
    self.minv2_2 = Lambda(FCModel_Functional.minv2, name="layer_minv2_2")([self.l2,self.v2]) 
    self.mergedOutput = Concatenate(axis=-1)([self.neutrini_layer,self.minv2_1,self.minv2_2])    
    self.model = tf.keras.models.Model(self.input_layer, self.mergedOutput, name=self.name)        
    #self.model = tf.keras.models.Model(self.input_layer, self.neutrini_layer, name=self.name)        
    print(self.model.summary())
    
    self.model.compile(**self.COMPILE_SETUP,loss=FCModel_Functional.customLoss(1E-7),run_eagerly=False) # it was 1E-8 when looking better
    #self.model.compile(**self.COMPILE_SETUP,run_eagerly=False)
