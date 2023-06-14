import os
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import random

class Model_Functional:
  SEED = 2022
  def __init__(self, name, files, output, setup, overrun={}, model=None):
    keras.utils.set_random_seed(Model_Functional.SEED)
    #tf.random.set_seed(Model_Functional.SEED)
    #random.seed(Model_Functional.SEED)

    self.model    = model
    self.setup    = setup
    self.output   = output
    self.log_dir  = self.output+"/log"
    self.files    = files    
    self.name     = name
    self.overrun  = overrun
    self.transformer = []
    for k in self.files:
      self.transformer.append(MinMaxScaler())
    #self.transformer = MinMaxScaler()
    self.CFG      = __import__(self.setup.replace('/', '.').strip('.py'), fromlist=[''])

    self.SETUP    = self.CFG.SETUP
    self.FEATURES = self.CFG.FEATURES

    self.target     = self.SETUP['target'     ] if 'target'     in self.SETUP.keys() else 'target'
    self.max_events = self.SETUP['max_events' ] if 'max_events' in self.SETUP.keys() else None

    assert self.target not in self.FEATURES , "The target variable is in the feature list"
    #assert not os.path.exists(self.output)  , "Output directory already exists"

    if not os.path.exists:
      os.makedirs(self.output)

  def predict(self, batch_size=500):
    for file in self.files:
      print("Running prediction on", file)
      path = self.output+"/"+os.path.basename(file)
      data = pd.read_hdf(file)[self.FEATURES]
      pred = self.model.predict(data, batch_size=batch_size)
      pd.DataFrame({'predictions': pred.reshape(len(pred))}).to_hdf(path, key='prediction')

  def predict(self, data):        
    pred = self.model.predict(data)
    #pred = self.transformer[0].inverse_transform(pred)    
    return pred
  
  def load(self):
    #for k in self.FEATURES:
    #  print(k)
    #print("from model.load file = ",self.files)
    self.inputs = {
      os.path.basename(h).strip('.h5'): pd.read_hdf(h).sample(frac=1, random_state=2022)[:self.max_events] for h in self.files
    } if self.max_events is not None else {
      os.path.basename(h).strip('.h5'): pd.read_hdf(h).sample(frac=1, random_state=2022) for h in self.files
    }
    normalization = 0.
    for k, v in self.inputs.items():
      print("samples ",k)
      if "DY" in k:
        normalization = v.shape[0]
        #print(normalization)

    for k, v in self.inputs.items():
      v["mGenReco"] =  v["mGenReco"]**2     
      v['sample'] = k               
      v["sample_weight"]=v["sample_weight"]*normalization/v.shape[0]
      
      if "ggF" in k:
        v["sample_weight"]=v["sample_weight"]
        v["sample_class1"]=1
      if "allEvents" in k:
        v["sample_weight"]=v["sample_weight"]
        v["sample_class1"]=1
      if "ggH" in k:
        v["sample_weight"]=v["sample_weight"]
        v["sample_class2"]=1
      if "DY" in k:
        v["sample_weight"]=v["sample_weight"]
        v["sample_class3"]=1
      if "TT" in k:
        v["sample_weight"]=v["sample_weight"]*1
        v["sample_class4"]=1
      #print(v["sample"].head(1) , v["sample_weight"].head(1))

    self.dframe = pd.concat(self.inputs.values()).reset_index()

    self.sampleweight = self.dframe.loc[self.dframe['is_train']==1, "sample_weight"].to_numpy()
    self.x_train = self.dframe.loc[self.dframe['is_train']==1, self.FEATURES]
    self.x_valid = self.dframe.loc[self.dframe['is_valid']==1, self.FEATURES]
    self.x_test  = self.dframe.loc[self.dframe['is_test' ]==1, self.FEATURES]    
    self.y_trains = []
    self.y_valids = []
    self.y_tests = []
    n =0 
  
    for k, v in self.inputs.items():
      v['sample'] = k   
      """
      self.y_trains.append(self.transformer[n].fit_transform(v.loc[v['is_train']==1,self.target]))
      self.y_valids.append(self.transformer[n].transform(v.loc[v['is_valid']==1,self.target]))
      self.y_tests.append( self.transformer[n].transform(v.loc[v['is_test']==1, self.target]))
      n=n+1
    self.y_train = np.concatenate(self.y_trains)
    self.y_valid = np.concatenate(self.y_valids)
    self.y_test = np.concatenate(self.y_tests)
    """    
    self.y_train = self.dframe.loc[self.dframe['is_train']==1, self.target  ]
    self.y_valid = self.dframe.loc[self.dframe['is_valid']==1, self.target  ]
    self.y_test  = self.dframe.loc[self.dframe['is_test' ]==1, self.target  ]    
  
  def fit(self, callbacks=[]):
    self.FIT_SETUP = self.SETUP['FIT']
    self.model.fit(
      self.x_train, self.y_train                    ,
      validation_data = [self.x_valid, self.y_valid],
      sample_weight=self.sampleweight               , 
      callbacks       = callbacks                   ,
      **self.FIT_SETUP
    )

    self.model.save(self.output)
    """
    self.model_json = self.model.to_json()
    with open(self.output, "w") as json_file:
      json_file.write(self.model_json)
    # serialize weights to HDF5
    self.model.save_weights(self.output)
    """ 
  @staticmethod
  def OVERRUN(ori, rep):
    for k, v in rep.items():
      if type(v)==dict: OVERRUN(ori[k], rep[k])
      ori[k] = rep[k]
