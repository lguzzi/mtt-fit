import os
import sys ; sys.path.append(os.getcwd())
import argparse
import numpy as np
from tensorflow   import keras
from models.Model_ssww import Model_Functional as Model
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats as st

 


parser = argparse.ArgumentParser('Compute predictions for a given model within the mTT-fit framework')
parser.add_argument('-i', '--input'     , required=True, nargs='+', help='List of input .h5 files (space separated)')
parser.add_argument('-o', '--output'    , default="testPlots"           , help='Directory used to store preidcions'       )
parser.add_argument('-s', '--setup'     , required=True           , help='Load the setup from a python script'      )
parser.add_argument('-m', '--model'     , required=True           , help='Path to the keras model'                  )
parser.add_argument('-d', '--draw'      , default=None           , help='Path to the keras model'                  )
args = parser.parse_args()

def recoMass(model):
  data = model.x_test
  neutrini = model.y_test
  #print(neutrini)
  #print(model.FEATURES)
  #data = data
  #print(data.columns)
  delta = pd.DataFrame()
  nu_predict = model.predict(data)

  data["neutrino1_px"] = nu_predict[:,1]
  data["neutrino1_py"] = nu_predict[:,2]
  data["neutrino1_pz"] = nu_predict[:,3]    
  #data["neutrino1_px"] = neutrini["pxv1"]
  #data["neutrino1_py"] = neutrini["pyv1"]
  #data["neutrino1_pz"] = neutrini["pzv1"]
  #data["neutrino1_e"] = np.sqrt(pow((data["neutrino1_px"]),2)+pow(data["neutrino1_py"],2)+pow(data["neutrino1_pz"],2))
  data["neutrino1_e"] = nu_predict[:,0]
  
  data["neutrino2_px"] = nu_predict[:,5]
  data["neutrino2_py"] = nu_predict[:,6]
  data["neutrino2_pz"] = nu_predict[:,7]
  #data["neutrino2_px"] = neutrini["pxv2"]
  #data["neutrino2_py"] = neutrini["pyv2"]
  #data["neutrino2_pz"] = neutrini["pzv2"]
  #data["neutrino2_e"] = np.sqrt(pow((data["neutrino2_px"]),2)+pow(data["neutrino2_py"],2)+pow(data["neutrino2_pz"],2))
  data["neutrino2_e"] = nu_predict[:,4]
  

  delta["px1"] = (data["neutrino1_px"] - neutrini["pxv1"])/(neutrini["pxv1"]+0.0000001)
  delta["py1"] = (data["neutrino1_py"] - neutrini["pyv1"])/(neutrini["pyv1"]+0.0000001)
  delta["pz1"] = (data["neutrino1_pz"] - neutrini["pzv1"])/(neutrini["pzv1"]+0.0000001)
  delta["px2"] = (data["neutrino2_px"] - neutrini["pxv2"])/(neutrini["pxv2"]+0.0000001)
  delta["py2"] = (data["neutrino2_py"] - neutrini["pyv2"])/(neutrini["pyv2"]+0.0000001)
  delta["pz2"] = (data["neutrino2_pz"] - neutrini["pzv2"])/(neutrini["pzv2"]+0.0000001)
  delta["nupx1"] = data["neutrino1_px"]
  delta["nupy1"] = data["neutrino1_py"]
  delta["nupz1"] = data["neutrino1_pz"]
  delta["nupe1"] = data["neutrino1_e"]
  delta["nupx2"] = data["neutrino2_px"]
  delta["nupy2"] = data["neutrino2_py"]
  delta["nupz2"] = data["neutrino2_pz"]
  delta["nupe2"] = data["neutrino2_e"]


  data["w11_px"] = data["pxl1"] + data["neutrino1_px"]
  data["w11_py"] = data["pyl1"] + data["neutrino1_py"]
  data["w11_pz"] = data["pzl1"] + data["neutrino1_pz"]
  data["w11_e"] = data["El1"] + data["neutrino1_e"]

  #data["w21_px"] = data["pxl2"] + data["neutrino1_px"]
  #data["w21_py"] = data["pyl2"] + data["neutrino1_py"]
  #data["w21_pz"] = data["pzl2"] + data["neutrino1_pz"]
  #data["w21_e"] = data["El2"] + data["neutrino1_e"]


  #data["w12_px"] = data["pxl1"] + data["neutrino2_px"]
  #data["w12_py"] = data["pyl1"] + data["neutrino2_py"]
  #data["w12_pz"] = data["pzl1"] + data["neutrino2_pz"]
  #data["w12_e"] = data["El1"] + data["neutrino2_e"]

  data["w22_px"] = data["pxl2"] + data["neutrino2_px"]
  data["w22_py"] = data["pyl2"] + data["neutrino2_py"]
  data["w22_pz"] = data["pzl2"] + data["neutrino2_pz"]
  data["w22_e"] = data["El2"] + data["neutrino2_e"]
  

  data["mw11"] = pow((data["w11_e"]),2) - pow((data["w11_px"]),2) - pow((data["w11_py"]),2) - pow((data["w11_pz"]),2)
  #data["mw21"] = pow((data["w21_e"]),2) - pow((data["w21_px"]),2) - pow((data["w21_py"]),2) - pow((data["w21_pz"]),2)
  #data["mw12"] = pow((data["w12_e"]),2) - pow((data["w12_px"]),2) - pow((data["w12_py"]),2) - pow((data["w12_pz"]),2)
  data["mw22"] = pow((data["w22_e"]),2) - pow((data["w22_px"]),2) - pow((data["w22_py"]),2) - pow((data["w22_pz"]),2)
  data_pos = data[data["mw11"] > 0]
  data_neg = data[data["mw11"] < 0]
  #data_neg["mw11"] = 0
  #data_pos_b = data[data["mw21"] > 0]
  #data_pos_c = data[data["mw12"] > 0]
  data_pos_d = data[data["mw22"] > 0]
  data_neg_d = data[data["mw22"] < 0]
  #data_neg_d["mw11"] = 0
  #sata_neg = data[data["mw11"] < 0]
  #data["mw11"] = pd.concat([np.sqrt(data_pos["mw11"]),data_neg_d["mw11"]],ignore_index=True)
  data["mw11"] = np.sqrt(data_pos["mw11"])
  #data["mw21"] = np.sqrt(data_pos_b["mw21"])
  #data["mw12"] = np.sqrt(data_pos_c["mw12"])
  #data["mw22"] = pd.concat([np.sqrt(data_pos_d["mw22"]),data_neg_d["mw11"]],ignore_index=True)
  data["mw22"] = np.sqrt(data_pos_d["mw22"])
  #mW1 = []
  #mW2 = []
  """
  for i,j,l,m in zip(data["mw11"].to_numpy(),data["mw12"].to_numpy(),data["mw21"].to_numpy(),data["mw22"].to_numpy()):
    if abs(i-80) < abs(j-80):
      mW1.append(i)
      mW2.append(m)
    else:
      mW1.append(j)
      mW2.append(l)
  """
  return data["mw11"].to_numpy(),data["mw22"].to_numpy(),delta
  
  

def recoMassOutput():
  class_true = [] 
  recomass = {}  
  for k in range(len(args.input)):  
    inputFile = args.input[k]    
    mymodel=args.model     

    model_TTsem = Model(
        name  = 'prediction wizard' , 
        files =  [inputFile],
        output= args.model          ,
        setup = args.setup          ,
        model = keras.models.load_model(mymodel,compile=False)
      )
    model_TTsem.load() 
    #print(model_TTsem.x_test.columns)
    output = recoMass(model_TTsem)    
    
  return output


print("Saved model ",args.model)

recomass = {}
recomass_even = recoMassOutput()



nbins=75
xmin = 0
xmax = 200
massFit = 0
massSigma = 0
bin_step=5
x = np.linspace(xmin,xmax,nbins)
fig = plt.figure(figsize=(10, 7), dpi=100)    
plt.xticks(fontsize=7,rotation = 90)
plt.grid(True,which='major',axis="x",linestyle=':')
plt.xticks(np.arange(xmin,xmax, bin_step))     
plt.hist(recomass_even[0],histtype="step",bins=nbins,range=(0,250),density=1,alpha=0.9)
plt.hist(recomass_even[1],histtype="step",bins=nbins,range=(0,250),density=1,alpha=0.9)
filename ="test_mW.png"
plt.savefig(args.output+"/"+filename)
for var in recomass_even[2].columns:
  fig = plt.figure(figsize=(10, 7), dpi=100)
  range = (-5,5)
  if "nu" in str(var):
    range = (-100,100)
  plt.hist(recomass_even[2][var],histtype="step",bins=nbins,range=range,density=1,alpha=0.9,label=var)
  plt.legend()
  plt.savefig(var+".png")


