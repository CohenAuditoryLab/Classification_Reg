import numpy as np
import scipy
import matplotlib.pyplot as plt
import pickle
filename='LFP_Raster_ToneIntensity_AC_wdw_start-125_dt200_training_results'
with open('./'+filename+'.pkl','rb') as f:
    FireRate_testR2, FireRate_validR2, FireRate_y_test_pred, FireRate_y_valid_pred, FireRate_best_params, FireRate_y_valid, FireRate_y_test,\
    LPF_testR2, LPF_validR2, LPF_y_test_pred, LPF_y_valid_pred, LPF_best_params, LPF_y_valid, LPF_y_test,\
    Cat_testR2, Cat_validR2, Cat_y_test_pred, Cat_y_valid_pred, Cat_best_params, Cat_y_valid, Cat_y_test=pickle.load(f)


Modcolors=['b','g']
Models=['LSTM','GRU']
fig = plt.figure(figsize=(18,10))
ax=fig.add_subplot(3,1,1)

x=range(len(FireRate_y_test))
ax.plot(x,FireRate_y_test,figure=fig,lw=2,c='r',label='Real')
for i in range(len(FireRate_testR2)):
    # ax.plot(x,FireRate_y_valid_pred[i],ls=':',lw=2,c=Modcolors[i],label=Models[i]+'_valid_pred'+'R2:'+str(FireRate_validR2[i]))
    labels=Models[i]+'_testR2:'+str(np.round(FireRate_testR2[i],decimals=2))+'_validR2:'+str(np.round(FireRate_validR2[i],decimals=2))
    ax.plot(x,FireRate_y_test_pred[i],lw=1,c=Modcolors[i],label=labels)
    ax.legend(loc=1)
    ax.set_title('Firerate')
    plt.ylabel('TonedB') 

ax=fig.add_subplot(3,1,2)
x=range(len(LPF_y_test))
ax.plot(x,LPF_y_test,figure=fig,lw=2,c='r',label='Real') 
for i in range(len(LPF_testR2)):
    # ax.plot(x,LPF_y_valid_pred[i],ls=':',lw=2,c=Modcolors[i],label=Models[i]+'_valid_pred'+'R2:'+str(LPF_validR2[i]))
    labels=Models[i]+'_testR2:'+str(np.round(LPF_testR2[i],decimals=2))+'_validR2:'+str(np.round(LPF_validR2[i],decimals=2))
    ax.plot(x,LPF_y_test_pred[i],lw=1,c=Modcolors[i],label=labels)
    ax.legend(loc=1)
    ax.set_title('LPF')
    plt.ylabel('TonedB') 

ax=fig.add_subplot(3,1,3)
x=range(len(Cat_y_test))
ax.plot(x,Cat_y_test,figure=fig,lw=2,c='r',label='Real') 
for i in range(len(Cat_testR2)):
    # ax.plot(x,Cat_y_valid_pred[i],ls=':',lw=2,c=Modcolors[i],label=Models[i]+'_valid_pred'+'R2:'+str(CAT_validR2[i]))
    labels=Models[i]+'_testR2:'+str(np.round(Cat_testR2[i],decimals=2))+'_validR2:'+str(np.round(Cat_validR2[i],decimals=2))  
    ax.plot(x,Cat_y_test_pred[i],lw=1,c=Modcolors[i],label=labels)
    ax.legend(loc=1)
    ax.set_title('CAT(firerate,LPF)')
    plt.ylabel('TonedB') 

plt.show()
fig.savefig(filename+'.png')
