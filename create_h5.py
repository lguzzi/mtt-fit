#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys ; assert sys.hexversion>=((3<<24)|(7<<16)), "Python 3.7 or greater required" # a binary joke is worth 111 words
sys.path.append(os.getcwd())

import ROOT
import pandas as pd, numpy as np
import math
import argparse
parser = argparse.ArgumentParser('Build .h5 files from a list of .root files for the HTT mass NN.\n\
NOTE: the test sample size is deduced from the --train-size and --valid-size arguments')
parser.add_argument('--input'     , required=True, nargs='+'  , help='List of the input .root file')
parser.add_argument('--output'    , required=True             , help='Output .h5 file')
parser.add_argument('--target'    , default='tauH_SVFIT_mass' , help='Target variable name')
parser.add_argument('--tree'      , default='HTauTauTree'     , help='Tree name')
parser.add_argument('--train-size', default=0.8 , type=float  , help='Fraction of the train sample')
parser.add_argument('--valid-size', default=0.2 , type=float  , help='Fraction of the validation sample')
parser.add_argument('--train_odd' , default=True , type=bool  , help='Use odd number of event for training')
parser.add_argument('--min'       , default=50  , type=float  , help='Min. value of the target variable')
parser.add_argument('--max'       , default=250 , type=float  , help='Max. value of the target variable')
parser.add_argument('--threads'   , default=1   , type=int    , help='Number of threads')
parser.add_argument('--features'  , required=True             , help='Python files with the FEATURES dictionary')
#parser.add_argument('--step'      , default=2   , type=float  , help='Step in the target variable used during flat-weight computation')
args = parser.parse_args()

ROOT.ROOT.EnableImplicitMT(args.threads)  
BRANCHES  = __import__(args.features.replace('/', '.').strip('.py'), fromlist=['']).BRANCHES
RBRANCHES = {k: (v, t) for k, (v, t) in BRANCHES.items() if not type(v) is type(lambda: None)}
LBRANCHES = {k: (v, t) for k, (v, t) in BRANCHES.items() if     type(v) is type(lambda: None)}

def train_test_valid_split(dframe, train_size, valid_size):
  dframe = dframe.sample(frac=1, random_state=2022).reset_index(drop=True)
  ti = math.ceil(len(dframe.index)*train_size)
  vi = math.ceil(len(dframe.index)*valid_size)+ti
  dframe.loc[  :ti-1, 'is_train'] = True
  dframe.loc[ti:vi-1, 'is_valid'] = True
  dframe.loc[vi:    , 'is_test' ] = True
  return dframe

def train_test_valid_split_eventNumber(dframe, train_size, valid_size, train_odd):
  dframe_test = []
  dframe_train = []
  if train_odd :
    dframe_test = dframe[dframe["eventNumber"].mod(2)==0].copy(deep=True)
    dframe_train = dframe[dframe["eventNumber"].mod(2)!=0].copy(deep=True)
  else:
    dframe_test = dframe[dframe["eventNumber"].mod(2)!=0].copy(deep=True)
    dframe_train = dframe[dframe["eventNumber"].mod(2)==0].copy(deep=True)

  dframe_train = dframe_train.sample(frac=1, random_state=2022).reset_index(drop=True)
  #setting even numbers for training and odd numbers for test. Then dividing further training sample in 80%,20%
  ti = math.ceil(len(dframe_train.index)*train_size)
  vi = math.ceil(len(dframe_train.index)*valid_size)+ti
  dframe_train.loc[  :ti-1, 'is_train'] = True
  dframe_train.loc[ti:vi-1, 'is_valid'] = True
  dframe_test.loc[0:,'is_test' ] = True
  dframe_tot = pd.concat([dframe_train,dframe_test])
  return dframe_tot

def flatten(dframe, target, min_t, max_t, step_t, subset, weight):
  bins = [x for x in range(min_t, max_t+step_t, step_t)]
  for ml, mh in zip(bins[:-1], bins[1:]):
    loc = (dframe[subset]==1) & (dframe[target]>=ml) & (dframe[target]<mh)
    dframe.loc[loc, weight] *= 1./len(dframe.loc[loc].index) if len(dframe.loc[loc].index) else 0
  dframe.loc[dframe[subset]==1, weight] *= 1./dframe.loc[dframe[subset]==1, weight].mean()

baseline = ' && '.join([
  'nleps==0'              ,
  'nbjetscand>1'          ,
  'isOS==1'               ,
  'dau2_deepTauVsJet>=5'  ,
  #'{T}>{m} && {T}<{M}'.format(T=args.target, m=args.min, M=args.max),
  'HHKin_mass>{m} && HHKin_mass<{M}'.format(m=args.min, M=args.max),
  '((pairType==0 && dau1_iso<0.15) || (pairType==1 && dau1_eleMVAiso==1) || (pairType==2 && dau1_deepTauVsJet>=5))',
])


iframe = ROOT.RDataFrame(args.tree, args.input)
iframe = iframe.Filter(baseline)
for k, (v, _) in RBRANCHES.items():
  iframe = iframe.Redefine(k, v) if k in iframe.GetColumnNames() else iframe.Define(k, v)
oframe = pd.DataFrame.from_dict(iframe.AsNumpy(columns=RBRANCHES.keys()))
for k, (_, t) in RBRANCHES.items():
  oframe[k] = oframe[k].astype(t) if not t is None else oframe[k]

for k, (v, t) in LBRANCHES.items():
  oframe[k] = oframe.apply(v, axis=1).astype(t) if not t is None else oframe.apply(v, axis=1)

oframe = oframe.drop(columns=['pairType'])
#oframe = train_test_valid_split(oframe, args.train_size, args.valid_size)
oframe = train_test_valid_split_eventNumber(oframe, args.train_size, args.valid_size,args.train_odd)
#flatten(oframe, args.target, args.min, args.max, args.step, 'is_train', 'sample_weight')
#flatten(oframe, args.target, args.min, args.max, args.step, 'is_valid', 'sample_weight')
#flatten(oframe, args.target, args.min, args.max, args.step, 'is_test' , 'sample_weight')

oframe.to_hdf(args.output, key='df')
