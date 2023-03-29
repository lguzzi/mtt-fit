import os
import sys ; sys.path.append(os.getcwd())
import argparse
import numpy as np
from tensorflow   import keras
from models.Model_Functional import Model_Functional as Model
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
  data["neutrino1_px"] = nu_predict[:,0]
  data["neutrino1_py"] = nu_predict[:,1]
  data["neutrino1_pz"] = nu_predict[:,2]
  data["neutrino1_e"] = nu_predict[:,3]
  data["neutrino2_px"] = nu_predict[:,4]
  data["neutrino2_py"] = nu_predict[:,5]
  data["neutrino2_pz"] = nu_predict[:,6]
  data["neutrino2_e"] = nu_predict[:,7]
  data["higgs_px"] = data["tau1_px"] + data["tau2_px"]+data["neutrino1_px"]+data["neutrino2_px"]
  data["higgs_py"] = data["tau1_py"] + data["tau2_py"]+data["neutrino1_py"]+data["neutrino2_py"]
  data["higgs_pz"] = data["tau1_pz"] + data["tau2_pz"]+data["neutrino1_pz"]+data["neutrino2_pz"]
  data["higgs_e"] = data["tau1_e"] + data["tau2_e"]+data["neutrino1_e"]+data["neutrino2_e"]

  data["mTauTauRegr"] = pow((data["higgs_e"]),2) - pow((data["higgs_px"]),2) - pow((data["higgs_py"]),2) - pow((data["higgs_pz"]),2)
  data_pos = data[data["mTauTauRegr"] > 0]
  #data_neg = data[data["mTauTauRegr"] < 0]
  data["mTauTauRegr"] = np.sqrt(data_pos["mTauTauRegr"])

  data["higgs_bb_px"] = data["bjet1_pt"].to_numpy() * np.cos(data["bjet1_phi"].to_numpy()) + data["bjet2_pt"].to_numpy() * np.cos(data["bjet2_phi"].to_numpy())
  data["higgs_bb_py"] = data["bjet1_pt"].to_numpy() * np.sin(data["bjet1_phi"].to_numpy()) + data["bjet2_pt"].to_numpy() * np.sin(data["bjet2_phi"].to_numpy())
  data["higgs_bb_pz"] = data["bjet1_pt"].to_numpy() * np.sinh(data["bjet1_eta"].to_numpy()) + data["bjet2_pt"].to_numpy() * np.sinh(data["bjet2_eta"].to_numpy())
  data["higgs_bb_e"] = data["bjet1_pt"].to_numpy() * np.cosh(data["bjet1_eta"].to_numpy()) + data["bjet2_pt"].to_numpy() * np.cosh(data["bjet2_eta"].to_numpy()) 

  data["mBB"] = pow((data["higgs_bb_e"]),2) - pow((data["higgs_bb_px"]),2) - pow((data["higgs_bb_py"]),2) - pow((data["higgs_bb_pz"]),2)
  data_posBB = data[data["mBB"] > 0]
  data["mBB"] = np.sqrt(data_posBB["mBB"])

  data["mHH"] = pow((data["higgs_bb_e"]+data["higgs_e"]),2) - pow((data["higgs_bb_px"]+data["higgs_px"]),2) - pow((data["higgs_bb_py"]+data["higgs_py"]),2) - pow((data["higgs_bb_pz"]+data["higgs_pz"]),2)
  data_posHH = data[data["mHH"] > 0]
  data["mHH"] = np.sqrt(data_posHH["mHH"])
  
  return output,data["mTauTauRegr"],data["mBB"],data["mHH"]
  
  

def recoMassOutput(sample="Odd"):
  class_true = [] 
  recomass = {}  
  model_train = "Even"
  if sample == "Even":
    model_train = "Odd"
  for k in range(len(args.input)):  
    inputFile = args.input[k]
    inputFile = inputFile.replace("train","train"+sample)
    mymodel=args.model 
    mymodel = mymodel.replace("train","train"+model_train)

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
    nEntries = len(output[0])
    if "ggF" in os.path.basename(args.input[k]):
      class_true = np.full(nEntries,0,int)
    if "DY" in os.path.basename(args.input[k]):
      class_true = np.full(nEntries,1,int)
    if "TTsem" in os.path.basename(args.input[k]):
      class_true = np.full(nEntries,2,int)
      
    myDF = np.array([output[0],output[1],output[2],output[3],class_true])
    myDF = np.transpose(myDF)      
    myDF = pd.DataFrame(data=myDF,index=model_TTsem.dframe.loc[model_TTsem.dframe['is_test' ]==1].index,columns=["discriminator","mTauTau","mBB","mHH","classTrue"])      
    recomass[os.path.basename(args.input[k]).strip('.h5').lstrip("output_")] =myDF
    variables = myDF.columns      

  return variables,recomass


print("Saved model ",args.model)

recomass = {}
variables, recomass_even = recoMassOutput("Even")   
_, recomass_odd = recoMassOutput("Odd")
#recomass_odd = recomass_even
class_true = []
class_pred = []
for k,v in recomass_even.items():
  recomass[k]=pd.concat([recomass_even[k], recomass_odd[k]],ignore_index=True)
  class_true = class_true+recomass[k]["classTrue"].to_numpy().tolist()
  class_pred = class_pred+recomass[k]["discriminator"].to_numpy().tolist()


from sklearn import metrics
fig = plt.figure(figsize=(10, 7), dpi=100)  
plt.set_cmap(plt.get_cmap("viridis"))
confusion_matrix = metrics.confusion_matrix(class_true, class_pred,normalize="true")
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["ggF","DY","TT"])
cm_display.plot()
filename ="ConfusionMatrix_"+str(args.model.split("/")[0])+".pdf"
plt.savefig(args.output+"/"+filename)

nbins=75
xmin = 0
xmax = 200
massFit = 0
massSigma = 0
bin_step=5
x = np.linspace(xmin,xmax,nbins)
for k,v in recomass.items():
  v_discr = v.loc[v["discriminator"]==0]  
  v_ell_regr = v.loc[pow((v["mTauTau"]-122)/25,2)+pow((v["mBB"]-111)/45,2)<1]
  #v_ell_regr = v.loc[pow((v["mTauTau"]-115)/25,2)+pow((v["mBB"]-111)/45,2)<1]
  v_discr_ell_regr = v_discr.loc[pow((v["mTauTau"]-122)/25,2)+pow((v["mBB"]-111)/45,2)<1]
  #v_discr_ell_regr = v_discr.loc[pow((v["mTauTau"]-115)/25,2)+pow((v["mBB"]-111)/45,2)<1]
  print("************** Sample ",k)
  print ("Total Numer of entries = ",len(v.index))
  print ("Discr == 0 Numer of entries = ",len(v_discr.index), " eff = ", round(len(v_discr.index)*1./len(v.index),2))
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


