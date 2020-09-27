#Import standard packages
import numpy as np
from sklearn import preprocessing


# Normalize LFP/firing rate data within each channel/neuron, save data recorded after sound onset
def Cut_Normalize(data, t, axis=2):
    # find the onset time of sound stimulus
    t_zero_ind=np.where(t==np.absolute(t).min())[1][0]
    data=data[:,t_zero_ind:,:]
    t=t[0,t_zero_ind:]
    # norm_data=np.empty_like(data)
    # for i in range(data.shape[axis]):
    #     norm_data[:,:,i]=preprocessing.StandardScaler().fit_transform(data[:,:,i])
    
    return data, t


# Generate firing rate from raster
def GenFirerate(raster,t, SmthWind=25):
    # get firing rates from raster
    Firerate =np.empty((raster.shape[0],raster.shape[1]-SmthWind+1,raster.shape[2]))
    for ii in range(raster.shape[2]):
        for jj in range(raster.shape[1]-SmthWind+1):
            Firerate[:,jj,ii]=np.sum(raster[:,jj:jj+SmthWind,ii],axis=1)/(SmthWind/1000)
    offsetInd=raster.shape[1]-SmthWind+1
    t=t[0,0:offsetInd].reshape(1,offsetInd)
    # reshape firerate if it is a one dimension array
    try:
        Firerate.shape[1]
    except IndexError:
        Firerate=Firerate.reshape(Firerate.shape[0],1)
    return Firerate, t

# # # bin input  
def Bin_input(inputs, input_times, wdw_end, wdw_start=-25, dt=200):
    #     input_times:  containing all the times of each datapoint
    # dt: size of time bins (ms)
    # wdw_start: the start time of the first bin relative to the first tone offset
    # wdw_end: the end time of the last bin, depending on release time, when sound stops
    BtoneInd=np.array([ 1,  2,  4,  5,  7,  8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29])
    OnEdges=[50*n+25*(n-1)+wdw_start for n in BtoneInd+1]
    OffEdges=[50*n+25*(n-1)+wdw_start+dt for n in BtoneInd+1]
    # # adjust num of data points for LFP data
    # if len(np.unique(np.diff(input_times)))>1:
    #     dt=int(np.ceil(dt/np.unique(np.diff(input_times))[0]))
    neural_data=np.empty([inputs.shape[0],len(BtoneInd),dt,inputs.shape[2]]) #Initialize array for binned neural data
    ToneOffTime=[50*n+25*(n-1) for n in BtoneInd+1]
    for i in range(len(BtoneInd)):
        tOn_temp=input_times-OnEdges[i]
        indOn=np.where(tOn_temp==min(k for k in tOn_temp if k >= 0))[0][0]
        indOff=indOn+dt
        neural_data[:,i,:,:]=inputs[:,indOn:indOff,:]
    
    # label all bins according to the sound on/off status
    Bins_OnOff=np.zeros([inputs.shape[0],len(BtoneInd)])
    for j in range(len(wdw_end)):
        Dtime=wdw_end[j]-ToneOffTime
        if max(Dtime)<0:
            continue
        else:
            binoff_ind=np.where(Dtime==min(k for k in Dtime if k >= 0))[0][0]
            Bins_OnOff[j,0:binoff_ind+1]=np.ones((binoff_ind+1))
    
    return neural_data, Bins_OnOff, wdw_start, dt
   


# Preprocess LFP data array, cut series around the target 
def PreprocessLFP(New_LFP,targetT,t_LFP,AllchannelLab,tbefTar=500,taftTar=500):
    LFP_cut=[]
    for j in range(len(targetT)):
        t_raster_temp=np.absolute(t_LFP-targetT[j])
        ind_temp=np.where(t_raster_temp==t_raster_temp.min())[0][0]
        LFP_cut.append(New_LFP[j,ind_temp-tbefTar:ind_temp+taftTar])
        # LFP_cut.append(New_LFP[j,ind_temp-tbefTar:ind_temp]-np.flipud(New_LFP[j,ind_temp+1:ind_temp+taftTar+1]))
    LFP_cut=np.array(LFP_cut)
    # z score LFPs recorded from the same channel,independently standardize each time point
    uniLab=np.unique(AllchannelLab)
    for kk in range(len(uniLab)):
        ind_temp=np.where(AllchannelLab==uniLab[kk])[0]
        LFP_temp=LFP_cut[ind_temp,:]
        if kk==0:
            Zscore_LFP=np.empty((len(AllchannelLab),LFP_cut.shape[1]))
        Zscore_LFP[ind_temp,:]=preprocessing.scale(LFP_cut[ind_temp,:])
        # Mean_LFP=np.mean(LFP_temp)
        # Std_LFP= np.std(LFP_temp,ddof=1)            
        # Zscore_LFP[ind_temp,:]=(LFP_cut[ind_temp,:]-Mean_LFP)/Std_LFP 
    # LFP_cut=preprocessing.scale(Zscore_LFP) # normalize input features 
    return Zscore_LFP

# Preprocess firing rate data array, cut series around the target 
def PreprocessSpike(New_Raster,targetT,AllneuronLab,semiToneDiff,tbefTar=0,taftTar=75,SmthWind=75):
    # cutting raster data
    Raster_cut=[]
    t_raster=np.array(range(New_Raster.shape[1]))-499
    for j in range(len(targetT)):
        ind_temp=np.nonzero(np.isin(t_raster,targetT[j]))[0][0]
        Raster_cut.append(New_Raster[j,ind_temp-tbefTar:ind_temp+taftTar])
    Raster_cut=np.array(Raster_cut)

    # get firing rates from raster_cut
    Firerate_temp1=[]
    Firerate_temp2=[]
    for ii in range(Raster_cut.shape[0]):
        for jj in range(Raster_cut.shape[1]-SmthWind+1):
            Firerate_temp1.append(np.sum(Raster_cut[ii,jj:jj+SmthWind])/(SmthWind/1000))
        Firerate_temp2.append(Firerate_temp1)       
        Firerate_temp1=[]
    Firerate=np.array(Firerate_temp2)
    # reshape firerate if it is a one dimension array
    try:
        Firerate.shape[1]
    except IndexError:
        Firerate=Firerate.reshape(Firerate.shape[0],1)

    # z score firing rates recorded from the same neuron
    if  np.any(semiToneDiff):
        combLabs = np.concatenate((AllneuronLab,semiToneDiff),axis=1)
    else:
        combLabs = AllneuronLab
    uniLab = np.unique(combLabs,axis=0)
    for kk in range(len(uniLab)):
        if np.any(semiToneDiff):
            ind_temp=np.array(np.where((combLabs[:,0] == uniLab[kk][0]) & (combLabs[:,1]==uniLab[kk][1])))
        else:
            ind_temp=np.array(np.where(combLabs[:,0] == uniLab[kk][0]))
        Firerate_temp=Firerate[ind_temp[0],:] 

        if kk==0:
            Zscore_Firerate=np.empty((len(AllneuronLab),Firerate.shape[1]))
        # Mean_Firate=np.mean(Firerate_temp)
        # Std_Firate= np.std(Firerate_temp,ddof=1)            
        # Zscore_Firerate[ind_temp[0],:]=(Firerate[ind_temp[0],:]-Mean_Firate)/Std_Firate; 
        Zscore_Firerate[ind_temp[0],:]=preprocessing.scale(Firerate[ind_temp[0],:])
    Zscore_Firerate[np.isnan(Zscore_Firerate)]=0
    # Firerate=preprocessing.scale(Zscore_Firerate) # normalize input features
    return Zscore_Firerate

# concatenate Spike and LFP features
def CAT_Spike_LFP(Firerate,LFP,channelLab_LPF,channelLab_Spike,trialNum_LPF,trialNum_Spike):
    CatFirerate_LFP=[]
    for i in range(Firerate.shape[0]):
        ind_temp=np.intersect1d(np.where(channelLab_LPF[:] == channelLab_Spike[i])[0],np.where(trialNum_LPF[:]==trialNum_Spike[i])[0])
        if len(ind_temp)!=1: 
            print(i)
        if Firerate.shape[1]==1:
            Firerate_temp=Firerate[i,:].reshape((1,))
        else:
            Firerate_temp=Firerate[i,:]
        CatFirerate_LFP.append(np.concatenate((Firerate_temp,np.ravel(LFP[ind_temp,:]))))
    CatFirerate_LFP=np.array(CatFirerate_LFP)
    return CatFirerate_LFP

# balance num of trial samples for different classes (hits vs misses/FAs)
def IndBalanceSamples(trialNum,y,posiLabel):
    uniComb= np.unique(np.concatenate((trialNum,y),axis=1),axis=0)
    clss,clssnum=np.unique(uniComb[:,1],return_counts=True)
    ind1=np.transpose(np.array(np.where(uniComb[:,1]==clss[0])))
    ind2=np.ravel(np.transpose(np.array(np.where(uniComb[:,1]==clss[posiLabel])))) #change the class for different dataset
    ind1=np.ravel((np.random.choice(np.squeeze(ind1),len(ind2),replace=False)).reshape(len(ind2),1))
    indpick1=np.intersect1d(np.where(np.isin(trialNum,uniComb[ind1,0]))[0], np.where(np.isin(y,uniComb[ind1,1]))[0])
    indpick2=np.intersect1d(np.where(np.isin(trialNum,uniComb[ind2,0]))[0], np.where(np.isin(y,uniComb[ind2,1]))[0])
    INDEX=np.concatenate((indpick1,indpick2))
    return INDEX




