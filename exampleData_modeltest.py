#Import standard packages
import sys
print(sys.version)
import time
from datetime import timedelta
start_time = time.monotonic()

from Decoding_models_hyperOpti import AllDecoders
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import stats
import pickle
from sklearn import preprocessing

#Import function to get the covariate matrix that includes spike history from previous bins
from preprocessing_funcs import get_spikes_with_history
#Import metrics
from metrics import get_R2
from metrics import get_rho

# load data
folder='./InputData/Decoding_Data/' #ENTER THE FOLDER THAT YOUR DATA IS IN
filename='example_data_m1'
with open(folder+filename+'.pickle','rb') as f:
    neural_data,vels_binned=pickle.load(f,encoding='latin1') #If using python 3

bins_before=6 #How many bins of neural data prior to the output are used for decoding
bins_current=1 #Whether to use concurrent time bin of neural data
bins_after=6 #How many bins of neural data after the output are used for decoding

# Format for recurrent neural networks (SimpleRNN, GRU, LSTM)
# Function to get the covariate matrix that includes spike history from previous bins
X=get_spikes_with_history(neural_data,bins_before,bins_after,bins_current)
#Set decoding output
y=vels_binned

FireRate_trainR2, FireRate_validR2, FireRate_testR2, FireRate_y_train_pred, FireRate_y_valid_pred, FireRate_y_test_pred, FireRate_y_train, FireRate_y_valid, FireRate_y_test, FireRate_best_params=AllDecoders(X,y)

with open(filename+'_training_results.pkl','wb') as f:
    pickle.dump([FireRate_trainR2, FireRate_validR2, FireRate_testR2, FireRate_y_train_pred, FireRate_y_valid_pred, FireRate_y_test_pred, FireRate_y_train, FireRate_y_valid, FireRate_y_test, FireRate_best_params],f)


print('Best param using firerate: ')
print(FireRate_best_params)


Modcolors=['b','g']
Models=['LSTM','GRU']
fig = plt.figure(figsize=(18,10))
ax=fig.add_subplot(2,1,1)
x=range(len(FireRate_y_test))
ax.plot(x,FireRate_y_test,figure=fig,lw=2,c='r',label='Real')
for i in range(len(FireRate_testR2)):
    labels=Models[i]+'_testR2:'+str(np.round(FireRate_testR2[i],decimals=2))+'_trainR2:'+str(np.round(FireRate_trainR2[i],decimals=2))    
    ax.plot(x,FireRate_y_test_pred[i][:,0],lw=1,c=Modcolors[i],label=labels)
    ax.legend(loc=1)
    plt.ylabel('X velocity') 

ax=fig.add_subplot(2,1,2)
x=range(len(FireRate_y_test))
ax.plot(x,FireRate_y_test,figure=fig,lw=2,c='r',label='Real')
for i in range(len(FireRate_testR2)):
    # ax.plot(x,FireRate_y_valid_pred[i],ls=':',lw=2,c=Modcolors[i],label=Models[i]+'_valid_pred'+'R2:'+str(FireRate_validR2[i]))
    labels=Models[i]+'_testR2:'+str(np.round(FireRate_testR2[i],decimals=2))+'_trainR2:'+str(np.round(FireRate_trainR2[i],decimals=2))    
    ax.plot(x,FireRate_y_test_pred[i][:,1],lw=1,c=Modcolors[i],label=labels)
    ax.legend(loc=1)
    plt.ylabel('Y velocity') 

fig.savefig(filename+'.png')
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
