#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys ; assert sys.hexversion>=((3<<24)|(7<<16)), "Python 3.7 or greater required" # a binary joke is worth 1000 words
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
parser.add_argument('--train-size', default=0.6 , type=float  , help='Fraction of the train sample')
parser.add_argument('--valid-size', default=0.3 , type=float  , help='Fraction of the validation sample')
parser.add_argument('--min'       , default=50  , type=float  , help='Min. value of the target variable')
parser.add_argument('--max'       , default=250 , type=float  , help='Max. value of the target variable')
parser.add_argument('--step'      , default=2   , type=float  , help='Step in the target variable used during flat-weight computation')
parser.add_argument('--threads'   , default=1   , type=int    , help='Number of threads')
parser.add_argument('--features'  , required=True             , help='Python files with the FEATURES dictionary')
args = parser.parse_args()

ROOT.ROOT.EnableImplicitMT(args.threads)
FEATURES  = __import__(args.features.replace('/', '.').strip('.py'), fromlist=['']).FEATURES
RFEATURES = {k: (v, t) for k, (v, t) in FEATURES.items() if not type(v) is type(lambda: None)}
LFEATURES = {k: (v, t) for k, (v, t) in FEATURES.items() if     type(v) is type(lambda: None)}

def train_test_valid_split(dframe, train_size, valid_size):
  dframe = dframe.sample(frac=1, random_state=2022).reset_index(drop=True)
  ti = math.ceil(len(dframe.index)*train_size)
  vi = math.ceil(len(dframe.index)*valid_size)+ti
  dframe.loc[  :ti-1, 'is_train'] = True
  dframe.loc[ti:vi-1, 'is_valid'] = True
  dframe.loc[vi:    , 'is_test' ] = True
  return dframe

def flatten(dframe, target, min_t, max_t, step_t, subset, weight):
  bins = [x for x in range(min_t, max_t+step_t, step_t)]
  for ml, mh in zip(bins[:-1], bins[1:]):
    loc = (dframe[subset]==1) & (dframe[target]>=ml) & (dframe[target]<mh)
    dframe.loc[loc, weight] *= 1./len(dframe.loc[loc].index) if len(dframe.loc[loc].index) else 0
  dframe.loc[dframe[subset]==1, weight] *= 1./dframe.loc[dframe[subset]==1, weight].mean()

baseline = ' && '.join([
  'nleps==0'              ,
  'nbjetscand>1'          ,
  'isOS!=0'               ,
  'dau2_deepTauVsJet>=5'  ,
  '{T}>{m} && {T}<{M}'.format(T=args.target, m=args.min, M=args.max),
  '((pairType==0 && dau1_iso<0.15) || (pairType==1 && dau1_eleMVAiso==1) || (pairType==2 && dau1_deepTauVsJet>=5))',
])

iframe = ROOT.RDataFrame(args.tree, args.input)
iframe = iframe.Filter(baseline)
for k, (v, _) in RFEATURES.items():
  iframe = iframe.Redefine(k, v) if k in iframe.GetColumnNames() else iframe.Define(k, v)
oframe = pd.DataFrame.from_dict(iframe.AsNumpy(columns=RFEATURES.keys()))
for k, (_, t) in RFEATURES.items():
  oframe[k] = oframe[k].astype(t) if not t is None else oframe[k]

for k, (v, t) in LFEATURES.items():
  oframe[k] = oframe.apply(v, axis=1).astype(t) if not t is None else oframe.apply(v, axis=1)

oframe = oframe.drop(columns=['pairType'])
oframe = train_test_valid_split(oframe, args.train_size, args.valid_size)
#flatten(oframe, args.target, args.min, args.max, args.step, 'is_train', 'sample_weight')
#flatten(oframe, args.target, args.min, args.max, args.step, 'is_valid', 'sample_weight')
#flatten(oframe, args.target, args.min, args.max, args.step, 'is_test' , 'sample_weight')

oframe.to_hdf(args.output, key='df')
