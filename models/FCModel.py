from keras                  import Sequential
from keras.layers           import InputLayer, Dense, Dropout

from models.Model import Model

class FCModel(Model):
  def __init__(self, dropout, neurons, **kwargs):
    super().__init__(**kwargs)
    self.dropout = dropout
    self.neurons = neurons

  def compile(self):
    self.DENSE_SETUP   = self.SETUP['DENSE'  ]
    self.LAST_SETUP    = self.SETUP['LAST'   ]
    self.COMPILE_SETUP = self.SETUP['COMPILE']

    self.model = Sequential(InputLayer(input_shape=len(self.FEATURES)), name=self.name)
    for n in self.neurons:
      self.model.add(Dense(n, **self.DENSE_SETUP))
      if self.dropout is not None:
        self.model.add(Dropout(self.dropout))
    self.model.add(Dense(1, **self.LAST_SETUP))
    
    print(self.model.summary())
    
    self.model.compile(**self.COMPILE_SETUP)
