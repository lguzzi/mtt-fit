import os
import sys ; sys.path.append(os.getcwd())
import argparse
import numpy as np
from tensorflow   import keras
from models.Model_Functional import Model_Functional as Model
from models.FCModel_Functional import FCModel_Functional
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
  #data = data
  #print(data.columns)

  nu_predict = model.predict(data)
  classifier = nu_predict[:,9:]
  output=[]
  for i in classifier:
    output.append(int(np.argmax(i)))
  
  higgs_bb_px = data["bjet1_pt"].to_numpy() * np.cos(data["bjet1_phi"].to_numpy()) + data["bjet2_pt"].to_numpy() * np.cos(data["bjet2_phi"].to_numpy())
  higgs_bb_py = data["bjet1_pt"].to_numpy() * np.sin(data["bjet1_phi"].to_numpy()) + data["bjet2_pt"].to_numpy() * np.sin(data["bjet2_phi"].to_numpy())
  higgs_bb_pz = data["bjet1_pt"].to_numpy() * np.sinh(data["bjet1_eta"].to_numpy()) + data["bjet2_pt"].to_numpy() * np.sinh(data["bjet2_eta"].to_numpy())
  higgs_bb_e = data["bjet1_pt"].to_numpy() * np.cosh(data["bjet1_eta"].to_numpy()) + data["bjet2_pt"].to_numpy() * np.cosh(data["bjet2_eta"].to_numpy()) 
  
  mass_bb = pow((higgs_bb_e),2) - pow((higgs_bb_px),2) - pow((higgs_bb_py),2) - pow((higgs_bb_pz),2)

  higgs_px = data["tau1_px"].to_numpy() + data["tau2_px"].to_numpy()
  higgs_py = data["tau1_py"].to_numpy() + data["tau2_py"].to_numpy()
  higgs_pz = data["tau1_pz"].to_numpy() + data["tau2_pz"].to_numpy()
  higgs_e= data["tau1_e"].to_numpy() + data["tau2_e"].to_numpy()
  
  for i in range(len(higgs_px)):    
    higgs_px[i] = higgs_px[i] + nu_predict[i][0] + nu_predict[i][4]
    higgs_py[i] = higgs_py[i] + nu_predict[i][1] + nu_predict[i][5]
    higgs_pz[i] = higgs_pz[i] + nu_predict[i][2] + nu_predict[i][6]
    higgs_e[i] = higgs_e[i] + nu_predict[i][3] + nu_predict[i][7]
  mass_tautau = pow((higgs_e),2) - pow((higgs_px),2) - pow((higgs_py),2) - pow((higgs_pz),2)
  mass_hh = pow((higgs_bb_e + higgs_e),2) - pow((higgs_bb_px + higgs_px),2) - pow((higgs_bb_py + higgs_py),2) - pow((higgs_bb_pz + higgs_pz),2)
  for i in range(len(mass_hh)):
    if mass_hh[i] <0:
      mass_hh[i] = 0
    if mass_tautau[i] <0:
      mass_tautau[i] =0
    if mass_bb[i] <0:
      mass_bb[i] =0

  mass_hh = np.sqrt(mass_hh)
  mass_tautau = np.sqrt(mass_tautau)
  mass_bb = np.sqrt(mass_bb)

  #minv = np.sqrt(mass2)
  return output,mass_tautau,mass_bb,mass_hh
  #return minv,output

print("Saved model ",args.model)

recomass = {}
variables = 0
for k in range(len(args.input)):      
      model_TTsem = Model(
          name  = 'prediction wizard' , 
          files =  [args.input[k]],
          output= args.model          ,
          setup = args.setup          ,
          model = keras.models.load_model(args.model,{"customMAE":None})
        )
      model_TTsem.load() 
      output = recoMass(model_TTsem)
      myDF = np.array([output[0],output[1],output[2],output[3]])
      myDF = np.transpose(myDF)      
      myDF = pd.DataFrame(data=myDF,index=model_TTsem.dframe.loc[model_TTsem.dframe['is_test' ]==1].index,columns=["discriminator","mTauTau","mBB","mHH"])
      myDF["tauH_SVFIT_mass"]=model_TTsem.dframe.loc[model_TTsem.dframe['is_test']==1,"tauH_SVFIT_mass"]
      myDF["mGenReco"]=np.sqrt(model_TTsem.dframe.loc[model_TTsem.dframe['is_test']==1,"mGenReco"])
      recomass[os.path.basename(args.input[k]).strip('.h5').lstrip("output_")] =myDF      
      variables = myDF.columns


nbins=75
xmin = 0
xmax = 200
massFit = 0
massSigma = 0
bin_step=5
x = np.linspace(xmin,xmax,nbins)
for k,v in recomass.items():
  v_discr = v.loc[v["discriminator"]==0]
  v_ell = v.loc[pow((v["tauH_SVFIT_mass"]-116)/35,2)+pow((v["mBB"]-111)/45,2)<1]
  v_ell_regr = v.loc[pow((v["mTauTau"]-122)/25,2)+pow((v["mBB"]-111)/45,2)<1]
  v_discr_ell_regr = v_discr.loc[pow((v["mTauTau"]-122)/25,2)+pow((v["mBB"]-111)/45,2)<1]
  print("************** Sample ",k)
  print ("Total Numer of entries = ",len(v.index))
  print ("Discr == 0 Numer of entries = ",len(v_discr.index), " eff = ", round(len(v_discr.index)*1./len(v.index),2))
  print ("Ellipse Numer of entries = ",len(v_ell.index), " eff = ", round(len(v_ell.index)*1./len(v.index),2))
  print ("Ellipse Regr. Numer of entries = ",len(v_ell_regr.index), " eff = ", round(len(v_ell_regr.index)*1./len(v.index),2))
  print ("Discr ==0 && Ellipse Regr. Numer of entries = ",len(v_discr_ell_regr.index), " eff = ", round(len(v_discr_ell_regr.index)*1./len(v.index),2))
  for var in variables: 
    fig = plt.figure(figsize=(10, 7), dpi=100)    
    plt.xticks(fontsize=7,rotation = 90)
    plt.grid(True,which='major',axis="x",linestyle=':')
    plt.title(var)
    if "mHH" in str(var):
      xmax = 2000
      bin_step=50
      nbins=75
    elif "discr" in str(var):
      xmax=5      
      bin_step=1
      nbins=5  
    else:
      xmax=200
      bin_step=5   
      nbins=75    
    plt.xticks(np.arange(xmin,xmax, bin_step))    
    nH, bins, patches = plt.hist(v[var],density=1,alpha=0.2,histtype="stepfilled",color="b",linewidth=2.,bins=nbins,range=(xmin,xmax),label =k.split("_")[1])              
    nH, bins, patches = plt.hist(v[var],density=1,alpha=0.9,histtype="step",linewidth=1.,color="b",bins=nbins,range=(xmin,xmax))              
    nH, bins, patches = plt.hist(v_discr[var],density=1,alpha=0.2,histtype="stepfilled",linewidth=2.,color="g",bins=nbins,range=(xmin,xmax),label ="Selected Discr "+k.split("_")[0])          
    nH, bins, patches = plt.hist(v_discr[var],density=1,alpha=0.9,histtype="step",linewidth=1.,color="g",bins=nbins,range=(xmin,xmax))          
    nH, bins, patches = plt.hist(v_ell[var],density=1,alpha=0.2,histtype="stepfilled",linewidth=2.,color="r",bins=nbins,range=(xmin,xmax),label ="Selected Ell "+k.split("_")[1])                
    nH, bins, patches = plt.hist(v_ell[var],density=1,alpha=0.9,histtype="step",linewidth=1.,color="r",bins=nbins,range=(xmin,xmax))                
    nH, bins, patches = plt.hist(v_ell_regr[var],density=1,alpha=0.2,histtype="stepfilled",linewidth=2.,color="y",bins=nbins,range=(xmin,xmax),label ="Selected Ell Regr "+k.split("_")[0])                
    nH, bins, patches = plt.hist(v_ell_regr[var],density=1,alpha=0.9,histtype="step",linewidth=1.,color="y",bins=nbins,range=(xmin,xmax))                
    #nH, bins, patches = plt.hist(v_discr_ell_regr[var],density=1,alpha=0.2,histtype="stepfilled",linewidth=2.,bins=nbins,range=(xmin,xmax),label ="Selected Discr + Ell Regr "+k.split("_")[0])                
    plt.legend()      
    filename =str(k)+"_"+str(var)+"_"+str(args.model.split("/")[0])+".pdf"
    plt.savefig(args.output+"/"+filename)


