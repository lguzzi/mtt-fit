from tensorflow        import keras
from keras.layers      import InputLayer, Dense, Dropout
from keras             import Sequential
import pandas as pd
import os, sys
sys.path.append(os.getcwd())

import argparse
parser = argparse.ArgumentParser('Simple Dense Sequence NN for mTT inference.\n\
Pass different arguments in the form --some-setup arg1='"string_val"' args2=float_val (note the multiple quotes for strings)'
)
parser.add_argument('-i', '--input'     , required=True   , nargs='+'           , help='List of input .h5 files (space separated)'        )
parser.add_argument('-o', '--output'    , required=True                         , help='Output trained file'                              )
parser.add_argument('-s', '--setup'     , required=True                         , help='Setup python file containing the SETUP dictionary')
parser.add_argument('-m', '--max-events', default=300000                        , help='max number of events per input samples'           )
parser.add_argument('-d', '--dropout'   , default=None                          , help='use dropout for the hidden layers (None=no)'      )
parser.add_argument('-n', '--neurons'   , default=[100]*10, nargs='+', type=int , help='list of neuron numbers (per layer)'               )
parser.add_argument('-p', '--print'     , action='store_true'                   , help='print the model'                                  )
parser.add_argument('-D', '--dry-run'   , action='store_true'                   , help='Don\'t fit'                                       )
args = parser.parse_args()

SETUP = __import__(args.setup.replace('/', '.').strip('.py'), fromlist=['']).SETUP

DENSE_SETUP   = SETUP['DENSE'  ]
LAST_SETUP    = SETUP['LAST'   ]
COMPILE_SETUP = SETUP['COMPILE']
FIT_SETUP     = SETUP['FIT'    ]

inputs = {
  os.path.basename(h): pd.read_hdf(h)[:args.max_events] for h in args.input
} if args.max_events is not None else {
  os.path.basename(h): pd.read_hdf(h) for h in args.input
}
dframe  = pd.concat(inputs)
x_train = dframe.drop(columns='target').loc[dframe.drop(columns='target')['is_train']==1]
x_valid = dframe.drop(columns='target').loc[dframe.drop(columns='target')['is_valid']==1]
x_test  = dframe.drop(columns='target').loc[dframe.drop(columns='target')['is_test' ]==1]
y_train = dframe.loc[dframe['is_train'], 'target']
y_valid = dframe.loc[dframe['is_valid'], 'target']
y_test  = dframe.loc[dframe['is_test' ], 'target']

NN_model = Sequential(InputLayer(input_shape=len(x_train.columns)))
for n in args.neurons[1:]:
  NN_model.add(Dense(n, **DENSE_SETUP))
  if args.dropout is not None:
    NN_model.add(Dropout(args.dropout))

NN_model.add(Dense(1, **LAST_SETUP))
if args.print:
  print(NN_model.summary())
NN_model.compile(**COMPILE_SETUP)

if not args.dry_run:
  NN_model.fit(
    x_train, y_train,
    validation_data = [x_valid, y_valid],
    **FIT_SETUP
  )

NN_model_json = NN_model.save(args.output)
