import time
from datetime import timedelta
start_time = time.monotonic()
#Import standard packages
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from scipy import io
from scipy import stats
from more_itertools import distinct_combinations
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from scipy.io import savemat
import json 
#import customized packages
from classification_models_hyperOpti import allClassifiers
from Preprocessing import PreprocessLFP,PreprocessSpike,CAT_Spike_LFP,IndBalanceSamples
from _split import StratifiedGroupKfold

figNaStr='HvsFA'
cmXYlabels=['Hit','FA'] # confusion matrix plot xyticklabels
posiLabel=2 #the label of the positive class

folds=5 # folds for cross validation
Allconf_matrix_list={}
Allaccuracy_train={}
Allaccuracy_test={}
Allaverage_precision={}
datasetStr='nonAC_LFP_' #['nonAC_LFP','AC_LFP','nonAC_Spike','AC_Spike']

# # load, read, and preprocess input data
folder='./InputData/'
filename='LFP_RT_behave_nonAC_neuronwise'
data=io.loadmat(folder+filename+'.mat')  
variable=['New_LFP','New_targetTime','New_all_channel','t','New_trialNum','behave_data']
New_LFP=data[variable[0]]  
targetT_L=data[variable[1]]
AllchannelLab_L=np.squeeze(data[variable[2]])
t_raster=data[variable[3]]
New_trialNum_L=data[variable[4]]
Allyy=data[variable[5]]  
  
# process LFP data
LFP_cut=PreprocessLFP(New_LFP,targetT_L,t_raster,AllchannelLab_L)

# balance num of trial samples for different classes (hits vs misses/FAs)
index=IndBalanceSamples(trialNum=New_trialNum_L,y=Allyy,posiLabel=posiLabel)
New_trialNum_L=New_trialNum_L[index]
AllneuronLab=AllchannelLab_L[index]
AllXX=LFP_cut[index,:]
Allyy=np.ravel(Allyy[index])

# Training the model (train each neuron separatly)
NumNeu=1
strname=datasetStr+figNaStr
trainingResults={}
trainingResults['Allconf_matrix_list'],trainingResults['Allaccuracy_train'],\
    trainingResults['Allaccuracy_test'],trainingResults['Allaverage_precision']=\
        allClassifiers(AllXX,Allyy,AllneuronLab,NumNeu,posiLabel,cmXYlabels=cmXYlabels,kfoldLabels=New_trialNum_L,strname=strname)
# output model results
# a_file = open(strname+'_'+str(NumNeu)+'.json', 'w')
# json.dump(trainingResults, a_file)
# a_file.close()
dump(trainingResults,strname+'_'+str(NumNeu)+'.pkl')

# Training the model (pool all neurons into training)
uniNerLab=np.unique(AllneuronLab)
NumNeu=len(uniNerLab)
strname=datasetStr+figNaStr
trainingResults={}
trainingResults['Allconf_matrix_list'],trainingResults['Allaccuracy_train'],\
    trainingResults['Allaccuracy_test'],trainingResults['Allaverage_precision']=\
        allClassifiers(AllXX,Allyy,AllneuronLab,NumNeu,posiLabel,cmXYlabels=cmXYlabels,kfoldLabels=New_trialNum_L,strname=strname)
# output model results
# a_file = open(strname+'_'+str(NumNeu)+'.json', 'w')
# json.dump(trainingResults, a_file)
# a_file.close()
dump(trainingResults,strname+'_'+str(NumNeu)+'.pkl')

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))






