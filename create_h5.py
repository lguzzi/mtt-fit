#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys ; assert sys.hexversion>=((3<<24)|(7<<16)), "Python 3.7 or greater required" # a binary joke is worth 1000 words

import ROOT
import pandas as pd, numpy as np
import math
from glob import glob
import argparse
parser = argparse.ArgumentParser('Build .h5 files from a list of .root files for the HTT mass NN.\n\
NOTE: the test sample size is deduced from the --train-size and --valid-size arguments')
#parser.add_argument('--input'     , required=True             , help='Input .root file of glob pattern')
#parser.add_argument('--output'    , required=True             , help='Output .h5 file')
parser.add_argument('--target'    , default='tauH_SVFIT_mass' , help='Target variable name')
parser.add_argument('--tree'      , default='HTauTauTree'     , help='Tree name')
parser.add_argument('--train-size', default=0.6 , type=float  , help='Fraction of the train sample')
parser.add_argument('--valid-size', default=0.3 , type=float  , help='Fraction of the validation sample')
parser.add_argument('--min'       , default=50  , type=float  , help='Min. value of the target variable')
parser.add_argument('--max'       , default=250 , type=float  , help='Max. value of the target variable')
parser.add_argument('--step'      , default=2   , type=float  , help='Step in the target variable used during flat-weight computation')
parser.add_argument('--threads'   , default=1   , type=int    , help='Number of threads')
args = parser.parse_args()

args.input  = '/gwteraz/users/dzuolo/HHbbtautauAnalysis/SKIMMED_Legacy2017_19Feb2021/SKIM_GGHH_NLO_cHHH1_xs/output_0.root'
args.output = 'test.h5'

ROOT.ROOT.EnableImplicitMT(args.threads)

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

features = {
  # standalone definitions
  'tau1_pt'                   : ('dau1_pt'                                                , 'float16' ),
  'tau1_eta'                  : ('dau1_eta'                                               , 'float16' ),
  'tau1_phi'                  : ('dau1_phi'                                               , 'float16' ),
  'tau2_pt'                   : ('dau2_pt'                                                , 'float16' ),
  'tau2_eta'                  : ('dau2_eta'                                               , 'float16' ),
  'tau2_phi'                  : ('dau2_phi'                                               , 'float16' ),
  'MET_pt'                    : ('met_et'                                                 , 'float16' ),
  'MET_phi'                   : ('met_phi>=M_PI?met_phi-2.0*M_PI:met_phi'                 , 'float16' ),
  'MET_covXX'                 : ('met_cov00'                                              , 'float32' ),
  'MET_covXY'                 : ('met_cov01'                                              , 'float32' ),
  'MET_covYY'                 : ('met_cov11'                                              , 'float32' ),
  'PU_npvs'                   : ('npv'                                                    , 'int16'   ),
  'bjet1_pt'                  : ('bjet1_pt'                                               , 'float16' ),
  'bjet1_eta'                 : ('bjet1_eta'                                              , 'float16' ),
  'bjet1_phi'                 : ('bjet1_phi'                                              , 'float16' ),
  'bjet1_deepFlavor'          : ('bjet1_bID_deepFlavor'                                   , 'float16' ),
  'bjet2_pt'                  : ('bjet2_pt'                                               , 'float16' ),
  'bjet2_eta'                 : ('bjet2_eta'                                              , 'float16' ),
  'bjet2_phi'                 : ('bjet2_phi'                                              , 'float16' ),
  'bjet2_deepFlavor'          : ('bjet2_bID_deepFlavor'                                   , 'float16' ),
  'VBFjet1_pt'                : ('VBFjet1_pt>0?VBFjet1_pt:0'                              , 'float16' ),
  'VBFjet1_eta'               : ('VBFjet1_pt>0?VBFjet1_eta:0'                             , 'float16' ),
  'VBFjet1_phi'               : ('VBFjet1_pt>0?VBFjet1_phi:0'                             , 'float16' ),
  'VBFjet1_deepFlavor'        : ('VBFjet1_pt>0?VBFjet1_btag_deepFlavor:0'                 , 'float16' ),
  'VBFjet2_pt'                : ('VBFjet1_pt>0?VBFjet2_pt:0'                              , 'float16' ),
  'VBFjet2_eta'               : ('VBFjet1_pt>0?VBFjet2_eta:0'                             , 'float16' ),
  'VBFjet2_phi'               : ('VBFjet1_pt>0?VBFjet2_phi:0'                             , 'float16' ),
  'VBFjet2_deepFlavor'        : ('VBFjet1_pt>0?VBFjet2_btag_deepFlavor:0'                 , 'float16' ),
  'tauH_SVFIT_mass'           : ('tauH_SVFIT_mass'                                        , 'float32' ),
  'target'                    : ('tauH_SVFIT_mass'                                        , 'float32' ),
  'pairType'                  : ('pairType'                                               , 'int16'   ),
  'is_test'                   : ('false'                                                  , 'bool'    ),
  'is_train'                  : ('false'                                                  , 'bool'    ),
  'is_valid'                  : ('false'                                                  , 'bool'    ),
  'sample_weight'             : ('1'                                                      , 'float32' ),
  # dependent definitions
  'mT1'                       : ('sqrt(2*tau1_pt*MET_pt *(1-cos(tau1_phi-MET_phi )))'     , 'float16' ),
  'mT2'                       : ('sqrt(2*tau2_pt*MET_pt *(1-cos(tau2_phi-MET_phi )))'     , 'float16' ),
  'mTtt'                      : ('sqrt(2*tau1_pt*tau2_pt*(1-cos(tau1_phi-tau2_phi)))'     , 'float16' ),
  'mTtot'                     : ('sqrt(mT1*mT1+mT2*mT2+mTtt*mTtt)'                        , 'float16' ),
}

baseline = ' && '.join([
  'nleps==0'              ,
  'nbjetscand>1'          ,
  'isOS!=0'               ,
  'dau2_deepTauVsJet>=5'  ,
  '{T}>{m} && {T}<{M}'.format(T=args.target, m=args.min, M=args.max),
  '((pairType==0 && dau1_iso<0.15) || (pairType==1 && dau1_eleMVAiso==1) || (pairType==2 && dau1_deepTauVsJet>=5))',
])

iframe = ROOT.RDataFrame(args.tree, glob(args.input))
iframe = iframe.Filter(baseline)
for k, (v, _) in features.items():
  iframe = iframe.Redefine(k, v) if k in iframe.GetColumnNames() else iframe.Define(k, v)
oframe = pd.DataFrame.from_dict(iframe.AsNumpy(columns=features.keys()))
for k, (_, t) in features.items():
  oframe[k] = oframe[k].astype(t)

# definitions either not compatible with ROOT or too long
oframe["channel"]     = np.vectorize(lambda x: {0:'mt', 1:'et', 2:'tt'}[x])(oframe.pairType)
oframe["N_neutrinos"] = np.vectorize(lambda x: {'tt':2, 'mt':3, 'et':3, 'mm':4, 'em':4, 'ee':4}[x])(oframe['channel']).astype('int16')

oframe = oframe.drop(columns=['pairType'])
oframe = train_test_valid_split(oframe, args.train_size, args.valid_size)
#flatten(oframe, args.target, args.min, args.max, args.step, 'is_train', 'sample_weight')
#flatten(oframe, args.target, args.min, args.max, args.step, 'is_valid', 'sample_weight')
#flatten(oframe, args.target, args.min, args.max, args.step, 'is_test' , 'sample_weight')

oframe.to_hdf(args.output, key='df')
