import time
from datetime import timedelta
start_time = time.monotonic()

from scipy import io
import numpy as np

from Preprocessing import Cut_Normalize, GenFirerate, Bin_input
from Decoding_models_hyperOpti import AllDecoders
import matplotlib.pyplot as plt
import matplotlib
import pickle

# # load, read, and preprocess input data
folder='./InputData/'
filename='LFP_Raster_ToneIntensity_Belt'
data=io.loadmat(folder+filename+'.mat')  
variable=['New_Raster','target_time','semitone_diff','behav_ind',
    'channelLab_R','neuronLab_R','New_LFP','soundOffTime','t_raster','t_LFP','stim_db']

New_Raster=data[variable[0]]
target_time=data[variable[1]]
semitone_diff=data[variable[2]]
behav_ind=data[variable[3]]
channelLab_R=data[variable[4]]
neuronLab_R=data[variable[5]]
New_LFP=data[variable[6]]
soundOffTime=data[variable[7]]
t_raster=data[variable[8]]
t_LFP=data[variable[9]]
stim_db=data[variable[10]]

# process spike series
Firerate, t_raster=GenFirerate(New_Raster,t_raster)
Firerate, t_raster=Cut_Normalize(Firerate,t_raster)
Firerate_bins,BinOnOffF,wdw_start, dt=Bin_input(Firerate,t_raster,soundOffTime)
Firerate_bins_flatten=Firerate_bins.reshape(Firerate_bins.shape[0]*Firerate_bins.shape[1],Firerate_bins.shape[2],Firerate_bins.shape[3])
BinOnOffF_flatten=BinOnOffF.reshape(BinOnOffF.shape[0]*BinOnOffF.shape[1])
# process LFP data
New_LFP, t_LFP=Cut_Normalize(New_LFP,t_LFP)
LFP_bins,BinOnOffL,wdw_start, dt=Bin_input(New_LFP,t_LFP,soundOffTime)
LFP_bins_flatten=LFP_bins.reshape(LFP_bins.shape[0]*LFP_bins.shape[1],LFP_bins.shape[2],LFP_bins.shape[3])
BinOnOffL_flatten=BinOnOffL.reshape(BinOnOffL.shape[0]*BinOnOffL.shape[1])

# delete bins without sound
soundOn_Ind=np.where(BinOnOffF_flatten==1)[0]
Firerate_bins_flatten=Firerate_bins_flatten[soundOn_Ind,:,:]
LFP_bins_flatten=LFP_bins_flatten[soundOn_Ind,:,:]
# output 
BtoneInd=np.array([ 1,  2,  4,  5,  7,  8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29])
stim_db=stim_db[:,BtoneInd]
stim_db=stim_db.reshape(stim_db.shape[0]*stim_db.shape[1])
Y_stimDb=stim_db[soundOn_Ind]
Y_stimDb=Y_stimDb.reshape(Y_stimDb.shape[0],1)

# concatenate firing rate and LFP 
CatFirerate_LFP_bins=np.concatenate((Firerate_bins_flatten,LFP_bins_flatten),axis=2)



FireRate_trainR2, FireRate_validR2, FireRate_testR2, FireRate_y_train_pred, FireRate_y_valid_pred, FireRate_y_test_pred, FireRate_y_train, FireRate_y_valid, FireRate_y_test, FireRate_best_params=AllDecoders(Firerate_bins_flatten,Y_stimDb)
LPF_trainR2, LPF_validR2, LPF_testR2, LPF_y_train_pred, LPF_y_valid_pred, LPF_y_test_pred, LPF_y_train, LPF_y_valid, LPF_y_test, LPF_best_params=AllDecoders(LFP_bins_flatten,Y_stimDb)
Cat_trainR2, Cat_validR2, Cat_testR2, Cat_y_train_pred, Cat_y_valid_pred, Cat_y_test_pred, Cat_y_train, Cat_y_valid, Cat_y_test, Cat_best_params=AllDecoders(CatFirerate_LFP_bins,Y_stimDb)

with open(filename+'_wdw_start'+str(wdw_start)+'_dt'+str(dt)+'_training_results.pkl','wb') as f:
    pickle.dump([FireRate_trainR2, FireRate_validR2, FireRate_testR2, FireRate_y_train_pred, FireRate_y_valid_pred, FireRate_y_test_pred, FireRate_y_train, FireRate_y_valid, FireRate_y_test, FireRate_best_params,
    LPF_trainR2, LPF_validR2, LPF_testR2, LPF_y_train_pred, LPF_y_valid_pred, LPF_y_test_pred, LPF_y_train, LPF_y_valid, LPF_y_test, LPF_best_params,
    Cat_trainR2, Cat_validR2, Cat_testR2, Cat_y_train_pred, Cat_y_valid_pred, Cat_y_test_pred, Cat_y_train, Cat_y_valid, Cat_y_test, Cat_best_params],f)


print('Best param using firerate: ')
print(FireRate_best_params)
print('Best param using LFP: ')
print(LPF_best_params)
print('Best param using firerate&LFP: ')
print(Cat_best_params)

# FireRate_testR2=[0.2,0.3,0.8] 
# FireRate_validR2=[0.4,0.7,0.9]
# FireRate_y_test_pred=[]
# FireRate_y_test_pred.append([0.5*x for x in range(len(Y_stimDb))])
# FireRate_y_test_pred.append([0.52*x for x in range(len(Y_stimDb))])
# FireRate_y_test_pred.append([0.53*x for x in range(len(Y_stimDb))])
# FireRate_y_valid_pred=[]
# FireRate_y_valid_pred.append([0.7*x for x in range(len(Y_stimDb))])
# FireRate_y_valid_pred.append([0.72*x for x in range(len(Y_stimDb))])
# FireRate_y_valid_pred.append([0.73*x for x in range(len(Y_stimDb))])

Modcolors=['b','g']
Models=['LSTM','GRU']
fig = plt.figure(figsize=(18,10))
ax=fig.add_subplot(3,1,1)

x=range(len(FireRate_y_test))
ax.plot(x,FireRate_y_test,figure=fig,lw=2,c='r',label='Real')
for i in range(len(FireRate_testR2)):
    # ax.plot(x,FireRate_y_valid_pred[i],ls=':',lw=2,c=Modcolors[i],label=Models[i]+'_valid_pred'+'R2:'+str(FireRate_validR2[i]))
    labels=Models[i]+'_testR2:'+str(np.round(FireRate_testR2[i],decimals=2))+'_trainR2:'+str(np.round(FireRate_trainR2[i],decimals=2))
    ax.plot(x,FireRate_y_test_pred[i],lw=1,c=Modcolors[i],label=labels)
    ax.legend(loc=1)
    ax.set_title('Firerate')
    plt.ylabel('TonedB') 

ax=fig.add_subplot(3,1,2)
x=range(len(LPF_y_test))
ax.plot(x,LPF_y_test,figure=fig,lw=2,c='r',label='Real') 
for i in range(len(LPF_testR2)):
    # ax.plot(x,LPF_y_valid_pred[i],ls=':',lw=2,c=Modcolors[i],label=Models[i]+'_valid_pred'+'R2:'+str(LPF_validR2[i]))
    labels=Models[i]+'_testR2:'+str(np.round(LPF_testR2[i],decimals=2))+'_trainR2:'+str(np.round(LPF_trainR2[i],decimals=2))
    ax.plot(x,LPF_y_test_pred[i],lw=1,c=Modcolors[i],label=labels)
    ax.legend(loc=1)
    ax.set_title('LPF')
    plt.ylabel('TonedB') 

ax=fig.add_subplot(3,1,3)
x=range(len(Cat_y_test))
ax.plot(x,Cat_y_test,figure=fig,lw=2,c='r',label='Real') 
for i in range(len(Cat_testR2)):
    # ax.plot(x,Cat_y_valid_pred[i],ls=':',lw=2,c=Modcolors[i],label=Models[i]+'_valid_pred'+'R2:'+str(CAT_validR2[i]))
    labels=Models[i]+'_testR2:'+str(np.round(Cat_testR2[i],decimals=2))+'_trainR2:'+str(np.round(Cat_trainR2[i],decimals=2))  
    ax.plot(x,Cat_y_test_pred[i],lw=1,c=Modcolors[i],label=labels)
    ax.legend(loc=1)
    ax.set_title('CAT(firerate,LPF)')
    plt.ylabel('TonedB') 

fig.savefig(filename+'_R2'+'_wdw_start'+str(wdw_start)+'_dt'+str(dt)+'.png')
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))