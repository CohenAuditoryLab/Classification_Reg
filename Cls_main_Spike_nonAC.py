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
datasetStr='nonAC_Spike_' #['nonAC_LFP','AC_LFP','nonAC_Spike','AC_Spike']

# # load, read, and preprocess input data
folder='./InputData/'
filename='Spike_RT_behave_nonAC_neuronwise'
data=io.loadmat(folder+filename+'.mat')  
variable=['New_Raster','behave_data','targetT','neuron_lab','semiToneDiff','channel_lab','New_trialNum']
New_Raster=data[variable[0]]
Allyy=data[variable[1]] #Load behave_data as y
targetT_S=data[variable[2]]
AllneuronLab=data[variable[3]]
semiToneDiff=data[variable[4]]
AllchannelLab_S=data[variable[5]]
New_trialNum_S=data[variable[6]]

# process spike series
# semiToneDiff=[]
Firerate=PreprocessSpike(New_Raster,targetT_S,AllneuronLab,semiToneDiff)
   
# balance num of trial samples for different classes (hits vs misses/FAs)
index=IndBalanceSamples(trialNum=New_trialNum_S,y=Allyy,posiLabel=posiLabel)
New_trialNum_S=New_trialNum_S[index]
AllneuronLab=AllneuronLab[index]
AllXX=Firerate[index,:]
Allyy=np.ravel(Allyy[index])

# Training the model (train each neuron separatly)
NumNeu=1
strname=datasetStr+figNaStr
trainingResults={}
trainingResults['Allconf_matrix_list'],trainingResults['Allaccuracy_train'],\
    trainingResults['Allaccuracy_test'],trainingResults['Allaverage_precision']=\
        allClassifiers(AllXX,Allyy,AllneuronLab,NumNeu,posiLabel,cmXYlabels=cmXYlabels,kfoldLabels=New_trialNum_S,strname=strname)
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
        allClassifiers(AllXX,Allyy,AllneuronLab,NumNeu,posiLabel,cmXYlabels=cmXYlabels,kfoldLabels=New_trialNum_S,strname=strname)
# output model results
# a_file = open(strname+'_'+str(NumNeu)+'.json', 'w')
# json.dump(trainingResults, a_file)
# a_file.close()
dump(trainingResults,strname+'_'+str(NumNeu)+'.pkl')

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))






