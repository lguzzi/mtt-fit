import os
import sys ; sys.path.append(os.getcwd())
import argparse
import numpy as np
from tensorflow   import keras
from models.Model_Functional import Model_Functional as Model
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats as st
import ROOT



parser = argparse.ArgumentParser('Compute predictions for a given model within the mTT-fit framework')
parser.add_argument('-i', '--input'     , required=True, nargs='+', help='List of input .h5 files (space separated)')
parser.add_argument('-o', '--output'    , default="testPlots"           , help='Directory used to store preidcions'       )
parser.add_argument('-s', '--setup'     , required=True           , help='Load the setup from a python script'      )
parser.add_argument('-m', '--model'     , required=True           , help='Path to the keras model'                  )
parser.add_argument('-d', '--draw'      , default=None           , help='Path to the keras model'                  )
parser.add_argument('-N', '--name'      , default='mTT_fit'                     , help='Model name'                                 )
args = parser.parse_args()

def recoMass(model):
  data = model.x_test
  #data = data
  
  nu_predict = model.predict(data)
  classifier = nu_predict[:,1:]

  output=[]
  for i in classifier:
    output.append(int(np.argmax(i)))
  
  higgs_bb_px = data["bjet1_pt"].to_numpy() * np.cos(data["bjet1_phi"].to_numpy()) + data["bjet2_pt"].to_numpy() * np.cos(data["bjet2_phi"].to_numpy())
  higgs_bb_py = data["bjet1_pt"].to_numpy() * np.sin(data["bjet1_phi"].to_numpy()) + data["bjet2_pt"].to_numpy() * np.sin(data["bjet2_phi"].to_numpy())
  higgs_bb_pz = data["bjet1_pt"].to_numpy() * np.sinh(data["bjet1_eta"].to_numpy()) + data["bjet2_pt"].to_numpy() * np.sinh(data["bjet2_eta"].to_numpy())
  higgs_bb_e = data["bjet1_pt"].to_numpy() * np.cosh(data["bjet1_eta"].to_numpy()) + data["bjet2_pt"].to_numpy() * np.cosh(data["bjet2_eta"].to_numpy()) 
  
  mass_bb = pow((higgs_bb_e),2) - pow((higgs_bb_px),2) - pow((higgs_bb_py),2) - pow((higgs_bb_pz),2)

  for i in range(len(mass_bb)):
    if mass_bb[i] <0:
      mass_bb[i] =0
  mass_bb = np.sqrt(mass_bb)

  data["k_predict"] =nu_predict[:,0] 
  mass_tautau = data["k_predict"]*data["mVis"]

  #minv = np.sqrt(mass2)
  return output,mass_tautau,mass_bb
  #return minv,output

print("Saved model ",args.model)

recomass = {}
variables = 0
tautauMass_ggF = []
print("Loaded model from disk")
for k in range(len(args.input)):          
    model_TTsem = Model(
            name  = 'prediction wizard' , 
            files =  [args.input[k]],
            output= args.model          ,
            setup = args.setup          ,
            #model = keras.models.load_model(args.model,{"customMAE":None})            
            model = keras.models.load_model(args.model,compile=False)            
          )
    model_TTsem.load() 
    #print(model_TTsem.FEATURES)
    output = recoMass(model_TTsem)
    if "ggF_BSM_allEvents" in os.path.basename(args.input[k]).strip('.h5'):
      tautauMass_ggF = output[1]
    myDF = np.array([output[0],output[1],output[2]])
    myDF = np.transpose(myDF)      
    myDF = pd.DataFrame(data=myDF,index=model_TTsem.dframe.loc[model_TTsem.dframe['is_test' ]==1].index,columns=["discriminator","mTauTau","mBB"])
    myDF["tauH_SVFIT_mass"]=model_TTsem.dframe.loc[model_TTsem.dframe['is_test']==1,"tauH_SVFIT_mass"]
    myDF["mGenReco"]=np.sqrt(model_TTsem.dframe.loc[model_TTsem.dframe['is_test']==1,"mGenReco"])
    myDF["mVis"]=model_TTsem.dframe.loc[model_TTsem.dframe['is_test']==1,"mVis"]
    myDF['HHKin_mass']=model_TTsem.dframe.loc[model_TTsem.dframe['is_test']==1,"HHKin_mass"]
    recomass[os.path.basename(args.input[k]).strip('.h5').lstrip("output_")] =myDF      
    variables = myDF.columns
    #if "allEvents" in str(args.input[k]):
    #  data_correlation  = model_TTsem.x_test.corr()
    #  import seaborn as sn
    #  plt.subplots(figsize=(20,15))
    #  sn.heatmap(data_correlation,annot=False)
    #  plt.show()

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
    if "discr" in str(var):
      xmax=5      
      bin_step=1
      nbins=5  
    elif "HHKin" in str(var):
      xmax=2000      
      bin_step=100
      nbins=100  
    else:
      xmax=200
      bin_step=5   
      nbins=75    
    plt.xticks(np.arange(xmin,xmax, bin_step))     
    nH, bins, patches = plt.hist(v[var],density=0,alpha=0.2,histtype="stepfilled",color="b",linewidth=2.,bins=nbins,range=(xmin,xmax),label =k.split("_")[0])              
    nH, bins, patches = plt.hist(v[var],density=0,alpha=0.9,histtype="step",linewidth=1.,color="b",bins=nbins,range=(xmin,xmax))              
    nH, bins, patches = plt.hist(v_discr[var],density=0,alpha=0.2,histtype="stepfilled",linewidth=2.,color="g",bins=nbins,range=(xmin,xmax),label ="Selected Discr "+k.split("_")[0])          
    nH, bins, patches = plt.hist(v_discr[var],density=0,alpha=0.9,histtype="step",linewidth=1.,color="g",bins=nbins,range=(xmin,xmax))          
    #nH, bins, patches = plt.hist(v_ell[var],density=0,alpha=0.2,histtype="stepfilled",linewidth=2.,color="r",bins=nbins,range=(xmin,xmax),label ="Selected Ell "+k.split("_")[0])                
    #nH, bins, patches = plt.hist(v_ell[var],density=0,alpha=0.9,histtype="step",linewidth=1.,color="r",bins=nbins,range=(xmin,xmax))                
    #nH, bins, patches = plt.hist(v_ell_regr[var],density=0,alpha=0.2,histtype="stepfilled",linewidth=2.,color="k",bins=nbins,range=(xmin,xmax),label ="Selected Ell Regr "+k.split("_")[0])                
    #nH, bins, patches = plt.hist(v_ell_regr[var],density=0,alpha=0.9,histtype="step",linewidth=1.,color="k",bins=nbins,range=(xmin,xmax))                
    #nH, bins, patches = plt.hist(v_discr_ell_regr[var],density=0,alpha=0.2,histtype="stepfilled",linewidth=2.,bins=nbins,range=(xmin,xmax),label ="Selected Discr + Ell Regr "+k.split("_")[0])                
    plt.legend()      
    filename =str(k)+"_"+str(var)+"_"+str(args.model.split("/")[0])+".pdf"
    plt.savefig(args.output+"/"+filename)


c1 = ROOT.TCanvas("c1","",800,600)
hmass = ROOT.TH1F("hmass","",100,0.,200.)

for i in tautauMass_ggF:
  hmass.Fill(i)
hmass.Draw()
myFit = hmass.Fit("gaus","","R",110,135)
c1.SaveAs("c1.png")
