import tensorflow   as tf
#from tensorflow.keras.constraints          import max_norm
from tensorflow.keras.optimizers.schedules import ExponentialDecay

def customMAE(y_true,y_pred):
  deltaNeutrini = tf.keras.backend.mean(tf.math.abs(y_pred[:,:6]-y_true[:,:6]),axis=-1)
  return deltaNeutrini


BRANCHES = {
  # ROOT features
  'El1'                    : ("El1"                                        , 'float32' ),
  'pxl1'                   : ("pxl1"                                        , 'float32' ),
  'pyl1'                   : ("pyl1"                                        , 'float32' ),
  'pzl1'                   : ("pzl1"                                        , 'float32' ),
  'El2'                    : ("El2"                                        , 'float32' ),
  'pxl2'                   : ("pxl2"                                        , 'float32' ),
  'pyl2'                   : ("pyl2"                                        , 'float32' ),
  'pzl2'                   : ("pzl2"                                        , 'float32' ),
  'Ej1'                    : ("Ej1"                                        , 'float32' ),
  'pxj1'                   : ("pxj1"                                        , 'float32' ),
  'pyj1'                   : ("pyj1"                                        , 'float32' ),
  'pzj1'                   : ("pzj1"                                        , 'float32' ),
  'Ej2'                    : ("Ej2"                                        , 'float32' ),
  'pxj2'                   : ("pxj2"                                        , 'float32' ),
  'pyj2'                   : ("pyj2"                                        , 'float32' ),
  'pzj2'                   : ("pzj2"                                        , 'float32' ),
  'Ev1'                    : ("Ev1"                                        , 'float32' ),
  'pyv1'                   : ("pyv1"                                        , 'float32' ),
  'pxv1'                   : ("pxv1"                                        , 'float32' ),
  'pzv1'                   : ("pzv1"                                        , 'float32' ),
  'Ev2'                    : ("Ev2"                                        , 'float32' ),
  'pxv2'                   : ("pxv2"                                        , 'float32' ),
  'pyv2'                   : ("pyv2"                                        , 'float32' ),
  'pzv2'                   : ("pzv2"                                        , 'float32' ),
  'idl_W1'                 : ("idl_W1"                                        , 'float32' ),
  'idl_W2'                 : ("idl_W2"                                        , 'float32' ),
  'idv_W1'                 : ("idv_W1"                                        , 'float32' ),
  'idv_W2'                 : ("idv_W2"                                        , 'float32' ),
  'met'                    : ("met"                                         , 'float32' ),
  'metphi'                 : ("metphi"                                      , 'float32' ),
  'mjj'                    : ("mjj"                                         , 'float32' ),
  'etaj1'                  : ("etaj1"                                       , 'float32' ),
  'etaj2'                  : ("etaj2"                                       , 'float32' ),
  'mW'                     : ('81.0'                                                      , 'float32' ),
  'mW'                     : ('81.0'                                                      , 'float32' ),
  'is_test'                : ('false'                                                  , 'bool'    ),
  'is_train'               : ('false'                                                  , 'bool'    ),
  'is_valid'               : ('false'                                                  , 'bool'    ),
}


FEATURES = [
  b for b in BRANCHES.keys() if not b in [
    "Ev1","pxv1","pyv1","pzv1",
    "Ev2","pxv2","pyv2","pzv2",
    "mW","mW",
    "is_test","is_train","is_valid"
  ]
]
#FEATURES = [b for b in BRANCHES.keys()]
SETUP = {
  'max_events': 10000000,
  #'target'    : 'tauH_SVFIT_mass',
  #'target'    : ['genNu_px', 'genNu_py', 'genNu_pz', 'genNu_e'],
  'target'    : ["Ev1","pxv1","pyv1","pzv1","Ev2","pxv2","pyv2","pzv2","mW","mW"], 
  'DENSE':    {
    'activation'        : 'relu'          ,
    #'kernel_constraint' : max_norm(3)     ,
    'kernel_initializer': 'glorot_uniform',
    'kernel_regularizer':  tf.keras.regularizers.L2(1E-3)
  },
  'LAST':     {
    'activation': 'linear',
  },
  'COMPILE':  {
    #'loss'      : 'mean_absolute_error'   ,
    #'loss'      : 'mean_squared_error'   ,
    'optimizer' : tf.keras.optimizers.Adam(learning_rate=ExponentialDecay(
      initial_learning_rate = 5e-3    ,
      decay_steps           = 50*1500 ,
      decay_rate            = 0.2     )
    ),
    #'optimizer' : tf.keras.optimizers.Adam(learning_rate=1e-3), 
    'metrics'   : [customMAE]      ,
  },
  'FIT':      {
    'batch_size'          : 500,
    'epochs'              : 300 ,
    'shuffle'             : True,
    'verbose'             : True,
    'use_multiprocessing' : True,
  },
}
