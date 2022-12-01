import os
import sys ; sys.path.append(os.getcwd())

from itertools          import product

import tensorflow as tf
from tensorflow.keras.callbacks         import TensorBoard, ModelCheckpoint,EarlyStopping
from models.FCModel_Functional         import FCModel_Functional
from callbacks.plotting_p4Nu import mTTPlotCallback

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf

import argparse
parser = argparse.ArgumentParser('Simple Dense Sequence NN for mTT inference.\n\
Pass different arguments in the form --some-setup arg1='"string_val"' args2=float_val (note the multiple quotes for strings)'
)
parser.add_argument('-i', '--input'     , required=True   , nargs='+'           , help='List of input .h5 files (space separated)'  )
parser.add_argument('-o', '--output'    , required=True                         , help='Output trained file'                        )
parser.add_argument('-s', '--setup'     , required=True                         , help='Load the setup from a python script'        )
parser.add_argument('-d', '--dropout'   , default=None                          , help='use dropout for the hidden layers (None=no)')
parser.add_argument('-n', '--neurons'   , default=200, nargs='+', type=int , help='list of neuron numbers (per layer)'         )
parser.add_argument('-N', '--name'      , default='mTT_fit'                     , help='Model name'                                 )
parser.add_argument('-D', '--dry-run'   , action='store_true'                   , help='Don\'t fit'                                 )
parser.add_argument('-F', '--fast'      , action='store_true'                   , help='Don\'t run plotting callbacks'              )
parser.add_argument(      '--cpu'       , action='store_true'                   , help='Run on CPU'                                 )
args = parser.parse_args()

assert not os.path.exists(args.output), "Output folder alredy exists, will not proceed"

if args.cpu: os.environ['CUDA_VISIBLE_DEVICES']=''
model = FCModel_Functional(
  name    = args.name   ,
  files   = args.input  ,
  output  = args.output ,
  setup   = args.setup  ,
  neurons = args.neurons,
  dropout = args.dropout,
  #n_targets = 4         ,
  n_targets = 8         ,
)
model.load()
model.compile()

callbacks = [
  TensorBoard(
    log_dir = model.log_dir
    #,histogram_freq=1,profile_batch = 8
  ),
  ModelCheckpoint(
    filepath  = args.output+'/checkpoints/epoch_{epoch:02d}', 
    save_freq = 20
  ),
 EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=20,
    mode="min",
    restore_best_weights=True,
 ), 
]
"""
+[#channel binned
  mTTPlotCallback(name=ch                               , 
    data    = model.x_test[model.dframe['channel']==ch] , 
    target  = model.y_test[model.dframe['channel']==ch] , 
    log_dir = model.log_dir                             )
  for ch in model.dframe['channel'].unique()
]
"""
model.fit(callbacks=callbacks)
