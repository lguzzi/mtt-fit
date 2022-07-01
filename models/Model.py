import os
import pandas as pd
from tensorflow import keras

class Model:
  NORM = lambda x: (x-x.mean())/x.std()
  SEED = 2022
  def __init__(self, name, files, output, setup, model=None):
    keras.utils.set_random_seed(Model.SEED)

    self.model    = model
    self.setup    = setup
    self.output   = output
    self.log_dir  = self.output+"/log"
    self.files    = files
    self.name     = name

    self.SETUP    = __import__(self.setup.replace('/', '.').strip('.py'), fromlist=['']).SETUP
    self.FEATURES = __import__(self.setup.replace('/', '.').strip('.py'), fromlist=['']).FEATURES

    self.target     = self.SETUP['target'     ] if 'target'     in self.SETUP.keys() else 'target'
    self.max_events = self.SETUP['max_events' ] if 'max_events' in self.SETUP.keys() else None

    assert self.target not in self.FEATURES , "The target variable is in the feature list"
    assert not os.path.exists(self.output)  , "Output directory already exists"

    os.makedirs(self.output)
  
  def load(self, norm_function=NORM):
    self.inputs = {
      os.path.basename(h).strip('.h5'): pd.read_hdf(h).sample(frac=1, random_state=2022)[:self.max_events] for h in self.files
    } if self.max_events is not None else {
      os.path.basename(h).strip('.h5'): pd.read_hdf(h).sample(frac=1, random_state=2022) for h in self.files
    }
    for k, v in self.inputs.items():
      v['sample'] = k
    self.dframe = pd.concat(self.inputs.values()).reset_index()

    self.dframe[[x for x in self.FEATURES if self.dframe[x].dtype!='int16']] = norm_function(self.dframe[[x for x in self.FEATURES if self.dframe[x].dtype!='int16']])

    self.x_train = self.dframe.loc[self.dframe['is_train']==1, self.FEATURES]
    self.x_valid = self.dframe.loc[self.dframe['is_valid']==1, self.FEATURES]
    self.x_test  = self.dframe.loc[self.dframe['is_test' ]==1, self.FEATURES]
    self.y_train = self.dframe.loc[self.dframe['is_train']==1, self.target  ]
    self.y_valid = self.dframe.loc[self.dframe['is_valid']==1, self.target  ]
    self.y_test  = self.dframe.loc[self.dframe['is_test' ]==1, self.target  ]
  
  def fit(self, callbacks=[]):
    self.FIT_SETUP = self.SETUP['FIT']
    self.model.fit(
      self.x_train, self.y_train                    ,
      validation_data = [self.x_valid, self.y_valid],
      callbacks       = callbacks                   ,
      **self.FIT_SETUP
    )
    self.model.save(self.output)