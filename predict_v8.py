import os
import sys ; sys.path.append(os.getcwd())
import argparse
import numpy as np
from tensorflow   import keras
from models.Model_Functional import Model_Functional as Model
from models.FCModel_Functional import FCModel_Functional

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


parser = argparse.ArgumentParser('Compute predictions for a given model within the mTT-fit framework')
parser.add_argument('-i', '--input'     , required=True, nargs='+', help='List of input .h5 files (space separated)')
parser.add_argument('-o', '--output'    , default="testPlots"           , help='Directory used to store preidcions'       )
parser.add_argument('-s', '--setup'     , required=True           , help='Load the setup from a python script'      )
parser.add_argument('-m', '--model'     , required=True           , help='Path to the keras model'                  )
parser.add_argument('-d', '--draw'      , default=None           , help='Path to the keras model'                  )
args = parser.parse_args()

def customMAE(y_true,y_pred):
    #print("mass ",y_pred[:,8],y_true[:,8])    
    #return tf.keras.backend.mean(math_ops.squared_difference(y_pred[:,:8],y_true[:,:8]),axis=-1)
    deltaNeutrini = tf.keras.backend.mean(tf.math.abs(y_pred[:,:8]-y_true[:,:8]),axis=-1)
    deltaMass2 = tf.keras.backend.mean(tf.math.abs(y_pred[:,8]-y_true[:,8]),axis=-1)
    alpha = 0.01
    return deltaNeutrini + alpha*deltaMass2 

def gauss(x,amp,mu,sigma):  
  return amp*np.exp(-(x-mu)**2/(2*sigma**2))

def gaussExp(x,amp,mu,sigma,alpha,beta):  
  y = amp*np.exp(-(x-mu)**2/(2*sigma**2))+((x-mu)**2)*alpha*np.exp(-abs(x-mu)*beta) 
      
  return y

def recoMass(model):
  data = model.x_test
  #data = data
  #print(data.columns)

  nu_predict = model.predict(data)
  mass2 = nu_predict[:,8]
  classifier = nu_predict[:,9:]
  output=[]
  for i in classifier:
    output.append(np.argmax(i))

  higgs_px = data["tau1_px"].to_numpy() + data["tau2_px"].to_numpy()
  higgs_py = data["tau1_py"].to_numpy() + data["tau2_py"].to_numpy()
  higgs_pz = data["tau1_pz"].to_numpy() + data["tau2_pz"].to_numpy()
  higgs_e= data["tau1_e"].to_numpy() + data["tau2_e"].to_numpy()
  for i in range(len(higgs_px)):    
    higgs_px[i] = higgs_px[i] + nu_predict[i][0] + nu_predict[i][4]
    higgs_py[i] = higgs_py[i] + nu_predict[i][1] + nu_predict[i][5]
    higgs_pz[i] = higgs_pz[i] + nu_predict[i][2] + nu_predict[i][6]
    higgs_e[i] = higgs_e[i] + nu_predict[i][3] + nu_predict[i][7]

  mass_higgs2 = (higgs_e*higgs_e)-(higgs_px)*(higgs_px)-(higgs_py)*(higgs_py)-(higgs_pz)*(higgs_pz)
  for i in range(len(mass_higgs2)):
    if mass_higgs2[i] <0:
      mass_higgs2[i] = 0
  mass_higgs = np.sqrt(mass_higgs2)  
  #minv = np.sqrt(mass2)
  return mass_higgs,output
  #return minv,output

print("Saved model ",args.model)

recomass = {}
recomass2 = {}
recomass_sel ={}
class_true = []
class_pred = []
for k in range(len(args.input)):      
      model_TTsem = Model(
          name  = 'prediction wizard' , 
          files =  [args.input[k]],
          output= args.model          ,
          setup = args.setup          ,
          model = keras.models.load_model(args.model,{"customMAE":None})
        )
      model_TTsem.load() 
      myMass = recoMass(model_TTsem)
      recomass[os.path.basename(args.input[k]).strip('.h5')] = myMass[0]      
      recomass2[os.path.basename(args.input[k]).strip('.h5')] = myMass[1]
      recomass_sel[os.path.basename(args.input[k]).strip('.h5')] = []      
      for i in range(len(myMass[0])):   
        class_pred.append(myMass[1][i])  
        if "NMSSM" in os.path.basename(args.input[k]):
          class_true.append(0)   
        if "ggF" in os.path.basename(args.input[k]):
          class_true.append(0)
        if "ggH" in os.path.basename(args.input[k]):
          class_true.append(1)          
        if "DY" in os.path.basename(args.input[k]):
          class_true.append(2)
        if "TTsem" in os.path.basename(args.input[k]):
          class_true.append(3)
        if myMass[1][i] == 2:
          recomass_sel[os.path.basename(args.input[k]).strip('.h5')].append(myMass[0][i])
from sklearn import metrics
plt.set_cmap(plt.get_cmap("viridis"))
#confusion_matrix = metrics.confusion_matrix(class_true, class_pred,normalize="true")
#cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["ggF", "ggH","DY","TT"])
#cm_display.plot()
#plt.show()

fig = plt.figure(figsize=(10, 7), dpi=100)
nbins=100
xmin = 0
xmax = 200
massFit = 0
massSigma = 0
plt.xticks(np.arange(xmin,xmax, 5.0))
plt.xticks(fontsize=5,rotation = 90)
plt.grid(True,which='major',axis="x",linestyle=':')
x = np.linspace(xmin,xmax,nbins)
for k,v in recomass.items():
  nH, bins, patches = plt.hist(v,density=1,alpha=0.7,histtype="step",linewidth=1.,bins=nbins,range=(xmin,xmax),label =k.split("_")[1])    
  plt.legend()
  #if "ggF" in k:
    #popt, pcov = curve_fit(gauss,x,nH,p0=(0.035,120,20))
    #poptExp, pcov = curve_fit(gaussExp,x,nH,p0=(0.035,120,20,0.05,0.1))
    #print(k.split("_")[1]," mass = ",popt[1]," sigma = ",popt[2])
    #print(k.split("_")[1]," mass = ",poptExp[1]," sigma = ",poptExp[2])
    #massFit = poptExp[1]
    #massSigma = poptExp[2]
    #y = gaussExp(x, poptExp[0],poptExp[1],poptExp[2],poptExp[3],poptExp[4])
    #plt.plot(x,y, color="r", linestyle="dashed")
#for k,v in recomass2.items():
#  plt.hist(v,density=1,alpha=0.7,histtype="step",linewidth=1.,bins=nbins,range=(xmin,xmax),label =k.split("_")[1])    
#  plt.legend()
#plt.show()
filename ="mass_"+str(args.model.split("/")[0])+".pdf"
if args.draw :
  plt.savefig(args.output+"/"+filename)
fig = plt.figure(figsize=(10, 7), dpi=100)
for k,v in recomass_sel.items():
    plt.hist(v,density=1,alpha=0.7,histtype="step",linewidth=1.,bins=500,range=(0,400),label =k.split("_")[1])     
    plt.legend()
plt.show()
#fig = plt.figure(figsize=(10, 7), dpi=100)
#for k,v in recomass2.items():
#    plt.hist(v,density=1,alpha=0.7,histtype="step",linewidth=1.,bins=5,range=(0,5),label =k.split("_")[1])     
#    plt.legend()
#plt.show()

#Computing acceptance within +/- 2 sigma
#nSigma = 1.5
#for k,v in recomass.items():
#  nSel =0.
#  for m in v:    
#    if (m < massFit + nSigma*massSigma) and (m>massFit -nSigma*massSigma):
#      nSel = nSel+1
#  print("acceptance for ",k.split("_")[1], " = ",round(nSel/len(v),2))
