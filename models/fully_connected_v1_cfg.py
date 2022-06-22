import keras
from keras.constraints import max_norm

FEATURES = {
  # ROOT features
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
  ## dependent expressions
  'mT1'                       : ('sqrt(2*tau1_pt*MET_pt *(1-cos(tau1_phi-MET_phi )))'     , 'float16' ),
  'mT2'                       : ('sqrt(2*tau2_pt*MET_pt *(1-cos(tau2_phi-MET_phi )))'     , 'float16' ),
  'mTtt'                      : ('sqrt(2*tau1_pt*tau2_pt*(1-cos(tau1_phi-tau2_phi)))'     , 'float16' ),
  'mTtot'                     : ('sqrt(mT1*mT1+mT2*mT2+mTtt*mTtt)'                        , 'float16' ),
  # lambda features
  'channel'                   : (lambda x: {0:'mt', 1:'et', 2:'tt'}[x['pairType']]                       , None   ),
  'N_neutrinos'               : (lambda x: {'tt':2, 'mt':3, 'et':3, 'mm':4, 'em':4, 'ee':4}[x['channel']], 'int16'),
}

SETUP = {
  'DENSE':    {
    'activation'        : 'relu'          ,
    'kernel_constraint' : max_norm(3)     ,
    'kernel_initializer': 'glorot_uniform',
  },
  'LAST':     {
    'activation': 'linear',
  },
  'COMPILE':  {
    'loss'      : 'mean_absolute_error' ,
    'optimizer' : 'Adam'                , 
    'metrics'   : keras.metrics.mae     ,
  },
  'FIT':      {
    'batch_size'          : 100 ,
    'epochs'              : 1   ,
    'shuffle'             : True,
    'verbose'             : True,
    'use_multiprocessing' : True,
  },
}
