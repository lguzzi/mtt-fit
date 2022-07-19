import os
import sys ; sys.path.append(os.getcwd())

import argparse
parser = argparse.ArgumentParser('Compute predictions for a given model within the mTT-fit framework'
)
parser.add_argument('-i', '--input'     , required=True, nargs='+', help='List of input .h5 files (space separated)')
parser.add_argument('-o', '--output'    , required=True           , help='Directory used to store preidcions'       )
parser.add_argument('-s', '--setup'     , required=True           , help='Load the setup from a python script'      )
parser.add_argument('-m', '--model'     , required=True           , help='Path to the keras model'                  )
args = parser.parse_args()

assert not os.path.exists(args.output), "Output folder alredy exists, will not proceed"

from tensorflow   import keras
from models.Model import Model

model = Model(
  name  = 'prediction wizard' , 
  files = args.input          ,
  output= args.output         ,
  setup = args.setup          ,
  model = keras.models.load_model(args.model)
)

model.predict()
