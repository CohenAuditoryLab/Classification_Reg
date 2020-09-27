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

import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve,plot_precision_recall_curve, \
    average_precision_score,PrecisionRecallDisplay,confusion_matrix,plot_confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import GridSearchCV
from joblib import dump
# Import models 
from sklearn.linear_model import LogisticRegression    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
# figNaStr='HvsFA'
# cmXYlabels=['Hit','FA']# confusion matrix plot xyticklabels
# posiLabel=2#the label of the positive class

figNaStr='HvsM'
cmXYlabels=['Hit','Miss']# confusion matrix plot xyticklabels
posiLabel=1#the label of the positive class


# confusion matrix function labels
cmLabel=[0,posiLabel]
folds=5
# models infos
models=['LoRe','KNN','svm','XGBoost','RF']
classifiers = [
    LogisticRegression(class_weight='balanced',max_iter=2000),
    KNeighborsClassifier(leaf_size=5),
    SVC(class_weight='balanced'),
    XGBClassifier(booster='gbtree',verbosity=1,subsample=0.9,predictor='cpu_predictor',objective='binary:logistic'),
    RandomForestClassifier(class_weight='balanced')]
param_grid=[{'penalty':['l2'], 'C':np.logspace(-6, 0, 10)},
            {'n_neighbors':range(1, 21, 2),'metric':['euclidean', 'manhattan', 'minkowski']},
            {'C':np.logspace(-6, 0, 10),'kernel':['poly', 'rbf', 'sigmoid']},
            {'n_estimators':[10, 100, 1000],'eta':np.array(range(1,10,2))/10,'gamma':[i/10.0 for i in range(0,5)],'max_depth':range(3,10,2),'min_child_weight':range(1,6,2)},
            {'n_estimators':[10, 100, 1000],'min_samples_split':range(2,10,2),'max_features':['sqrt', 'log2']} ]

# load data
folder='./InputData/'
#filename='Spike_RT_behave_AC_neuronwise'
filename='LFP_RT_behave_AC_neuronwise'
data=io.loadmat(folder+filename+'.mat')  
variable=['New_LFP','behave_data','New_targetTime','New_all_channel','t']

# input
New_LFP=data[variable[0]] # load LFP
Allyy=data[variable[1]] #Load behave_data as y
targetT=data[variable[2]]
AllneuronLab=np.squeeze(data[variable[3]])
t_raster=data[variable[4]]

# cutting LFP data
LFP_cut=[]
t_raster=data[variable[4]]
tbefTar=500 #ms
taftTar=500 #ms
for j in range(len(targetT)):
    t_raster_temp=np.absolute(t_raster-targetT[j])
    ind_temp=np.where(t_raster_temp==t_raster_temp.min())[1][0]
    LFP_cut.append(New_LFP[j,ind_temp-tbefTar:ind_temp+taftTar])
    # LFP_cut.append(New_LFP[j,ind_temp-tbefTar:ind_temp]-np.flipud(New_LFP[j,ind_temp+1:ind_temp+taftTar+1]))

LFP_cut=np.array(LFP_cut)

# set input and output
AllXX=LFP_cut
Allyy=np.ravel(Allyy)

uniNerLab,Labcounts=np.unique(AllneuronLab,return_counts=True)
 
# train models with data from 1,2,3,... neurons in a iteration
NumNeu=1 
NerLab=list(distinct_combinations(list(uniNerLab),NumNeu))

# NumNeu=len(uniNerLab)
# NerLab=[tuple(uniNerLab)]

Allconf_matrix_list=[]
Allaccuracy_train=np.empty([len(NerLab),(len(models))])
Allaccuracy_test=np.empty([len(NerLab),(len(models))])
Allaverage_precision=np.empty([len(NerLab),(len(models))])
for i in range(len(NerLab)):
    ind=np.nonzero(np.isin(AllneuronLab,NerLab[i]))
    print('ALLsamples in neuron#'+ str(NerLab[i])+':'+str(len(ind[0])))
    AllX=AllXX[ind[0]]
    Ally=Allyy[ind[0]]
    # balance number of samples across classes, equal to the num of samples from the smallest class
    clss,clssnum=np.unique(Ally,return_counts=True)
    ind1=np.transpose(np.array(np.where(Ally==clss[0])))
    ind2=np.ravel(np.transpose(np.array(np.where(Ally==clss[posiLabel])))) #change the class for different dataset

    ind1=np.ravel((np.random.choice(np.squeeze(ind1),len(ind2),replace=False)).reshape(len(ind2),1))
    X=np.concatenate((np.take(AllX,ind1,axis=0),np.take(AllX,ind2,axis=0)),axis=0)
    y=np.concatenate((np.take(Ally,ind1,axis=0),np.take(Ally,ind2,axis=0)),axis=0)
    print('Balancedsamples in neuron#'+ str(NerLab[i])+':'+str(len(y)))
    # # initialize the arrays to save accuracy for each iteration in all models
    conf_matrix_list=[]
    accuracy_train=np.empty((len(models)))
    accuracy_test=np.empty((len(models)))
    average_precision=np.empty((len(models)))
    # zscore input and then normalize input: scale features to [0,1]
    #X=stats.zscore(X,axis=0)
    # X_norm=MinMaxScaler().fit_transform(np.concatenate((X.reshape(len(X),1),X.reshape(len(X),1)),axis=1)) 
    X_norm=MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2,stratify=y)

    # if (i==0):
    #     # for imbalanced dataset, calculate the null accuracy according to testing class distribution
    #     yIndex=pd.Index(y_valid.reshape(y_valid.shape[0]))
    #     print('Percent of each class in the validation set:(null accuracy is the largest %)')
    #     print(yIndex.value_counts(normalize=True))
    #     print('Num of trials in validating:')
    #     print(X_valid.shape[0])
    #     print('Num of trials in training:')
    #     print(X_train.shape[0])

    # iterate over classifiers
    k=0
    for param, clf in zip(param_grid, classifiers):
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator = clf, param_grid = param, cv = folds, n_jobs = -1, verbose = 0)
        # Fit the grid search to the data
        grid_search.fit(X_train, y_train)
        # dump(grid_search, models[k]+'gs_object_'+filename+'_'+variable[0]+'.pkl')
        # predict accuracy 
        accuracy_test[k]=grid_search.score(X_test,y_test)   
        accuracy_train[k]=grid_search.best_score_
        # plot PR curve
        if k in [0,2]:
            y_score=grid_search.decision_function(X_test)
            precision, recall, _ = precision_recall_curve(y_test, y_score, pos_label=posiLabel)
            average_precision[k] = average_precision_score(y_test, y_score, pos_label=posiLabel)
        if k in [1,3,4]:
            y_score=grid_search.predict_proba(X_test)
            precision, recall, _ = precision_recall_curve(y_test, y_score[:,1], pos_label=posiLabel)
            average_precision[k] = average_precision_score(y_test, y_score[:,1], pos_label=posiLabel)
        print("Average_precision of %s : %f" % (models[k],average_precision[k]))
        # plot confusion matrix
        conf_matrix = confusion_matrix(y_test, grid_search.predict(X_test),labels=cmLabel,normalize='true')
        conf_matrix_list.append(conf_matrix)  
        # summarize results
        print("Best mean training accuracy of %s : %f using %s" % (models[k],grid_search.best_score_, grid_search.best_params_))
        print("testing accuracy of %s : %f" % (models[k],accuracy_test[k]))
        k+=1
    Allconf_matrix_list.append(np.array(conf_matrix_list))
    Allaccuracy_train[i,:]= accuracy_train
    Allaccuracy_test[i,:]= accuracy_test
    Allaverage_precision[i,:]= average_precision
    
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))


if NumNeu==len(uniNerLab):
    fig1=plt.figure(1)
    ax1=plt.gca()
    x=np.arange(len(models))+1
    plt.plot(x,Allaccuracy_train[0,:],figure=fig1,marker='o',linestyle=None,label='TrainAcc')
    plt.plot(x,Allaccuracy_test[0,:],figure=fig1,marker='o',linestyle=None,label='TestAcc')
    fig1.suptitle('cross validation folds = '+str(folds))
    plt.ylim(0,1.1)
    plt.ylabel('Accuracy')
    ax1.legend()
    plt.xticks(x,models)
    fig1.savefig('AllAccuracy_'+filename+'_'+variable[0]+'_NumChannel'+str(NumNeu)+figNaStr+'.png')
    # plt.show()


    # plot the average_precision  
    fig2=plt.figure(2)
    ax2=plt.gca()
    x=np.arange(len(models))+1
    plt.plot(x,Allaverage_precision[0,:],figure=fig2,marker='o',linestyle=None)
    plt.ylim(0,1.1)
    plt.ylabel('Ave precision score')
    plt.xticks(x,models)
    fig2.suptitle('cross validation folds = '+str(folds))
    fig2.savefig('API_All_'+filename+'_'+variable[0]+'_NumChannel'+str(NumNeu)+figNaStr+'.png')
    # plt.show()

    # plot the confusion matrix  
    fig3 = plt.figure(3,figsize=(18,10))
    xx, yy = np.meshgrid([0,1], [0,1]) # generate ticks for text in matshow

    for u in range(len(models)):
        conf_matrix_temp=np.squeeze(np.array(Allconf_matrix_list)[0,u,:,:])
        ax = fig3.add_subplot(2,3,u+1)
        cax = ax.matshow(conf_matrix_temp,cmap=plt.cm.Blues,vmin=0, vmax=1)
        zz=conf_matrix_temp.flatten()
        for v, (x_val, y_val) in enumerate(zip(xx.flatten(),yy.flatten())):
            ax.text(x_val-0.1,y_val,round(zz[v],2),color='r',fontweight='bold',va='center', ha='center')
        plt.title('CM_'+models[u])   
        fig3.colorbar(cax,ax=ax)
        ax.set_xticklabels([''] + cmXYlabels)
        ax.set_yticklabels([''] + cmXYlabels)
        plt.xlabel('Predicted')
        plt.ylabel('True') 
    fig3.suptitle('Confusion matrix of all Classifiers'+'(Num of ChannelCombs = '+str(len(NerLab))+')')
    fig3.savefig('ConfMat_'+filename+'_'+variable[0]+'_NumChannel'+str(NumNeu)+figNaStr+'.png')
    # plt.show()
else:
    fig1=plt.figure(1)
    ax1=plt.gca()
    x=np.arange(len(models))+1
    train_mean=np.mean(Allaccuracy_train,axis=0).reshape(len(models),1)
    train_sem=np.array([np.std(Allaccuracy_train,axis=0)*np.sqrt(1./len(NerLab)+1./(len(NerLab)-1))]).reshape(len(models))
    plt.errorbar(x,train_mean,yerr=train_sem,figure=fig1,marker='o',linestyle=None,label='TrainAcc',capsize=0.5)
    test_mean=np.mean(Allaccuracy_test,axis=0).reshape(len(models),1)
    test_sem=np.array([np.std(Allaccuracy_test,axis=0)*np.sqrt(1./len(NerLab)+1./(len(NerLab)-1))]).reshape(len(models))
    plt.errorbar(x,test_mean,yerr=test_sem,figure=fig1,marker='s',linestyle=None,label='TestAcc',capsize=0.5)
    fig1.suptitle('Num of neurCombs = '+str(len(NerLab)))
    plt.ylim(0,1.1)
    plt.ylabel('Accuracy')
    ax1.legend()
    plt.xticks(x,models)
    fig1.savefig('AllAccuracy_'+filename+'_'+variable[0]+'_NumChannel'+str(NumNeu)+figNaStr+'.png')
    # plt.show()


    # plot the average_precision  
    fig2=plt.figure(2)
    ax2=plt.gca()
    x=np.arange(len(models))+1
    average_precision_mean=np.mean(Allaverage_precision,axis=0).reshape(len(models),1)
    average_precision_sem=stats.sem(Allaverage_precision,axis=0).reshape(len(models))
    plt.errorbar(x,average_precision_mean,yerr=average_precision_sem,figure=fig2,marker='o',linestyle=None,uplims=True,lolims=True)
    plt.ylim(0,1.1)
    plt.ylabel('Ave precision score')
    plt.xticks(x,models)
    fig2.suptitle('Num of neurCombs = '+str(len(NerLab)))
    fig2.savefig('API_All_'+filename+'_'+variable[0]+'_NumChannel'+str(NumNeu)+figNaStr+'.png')
    # plt.show()

    # plot the confusion matrix  
    fig3 = plt.figure(3,figsize=(18,10))
    xx, yy = np.meshgrid([0,1], [0,1]) # generate ticks for text in matshow

    for u in range(len(models)):
        conf_matrix_temp=np.squeeze(np.array(Allconf_matrix_list)[:,u,:,:])
        conf_matrix_temp_mean=np.squeeze(np.mean(conf_matrix_temp,axis=0))
        # conf_matrix_temp_mean=conf_matrix_temp
        conf_matrix_temp_sem=np.squeeze(stats.sem(conf_matrix_temp,axis=0))
        ax = fig3.add_subplot(2,3,u+1)
        cax = ax.matshow(conf_matrix_temp_mean,cmap=plt.cm.Blues,vmin=0, vmax=1)
        zz=conf_matrix_temp_mean.flatten()
        for v, (x_val, y_val) in enumerate(zip(xx.flatten(),yy.flatten())):
            ax.text(x_val-0.1,y_val,round(zz[v],2),color='r',fontweight='bold',va='center', ha='center')
            ax.text(x_val-0.1,y_val+0.1,'+/-',color='r',fontweight='bold',va='center', ha='center')
            ax.text(x_val-0.1,y_val+0.2,round(conf_matrix_temp_sem.flatten()[v],2),color='r',fontweight='bold',va='center', ha='center')

        plt.title('CM_'+models[u])   
        fig3.colorbar(cax,ax=ax)
        ax.set_xticklabels([''] + cmXYlabels)
        ax.set_yticklabels([''] + cmXYlabels)
        plt.xlabel('Predicted')
        plt.ylabel('True') 
    fig3.suptitle('Confusion matrix of all Classifiers'+'(Num of ChannelCombs = '+str(len(NerLab))+')')
    fig3.savefig('ConfMat_'+filename+'_'+variable[0]+'_NumChannel'+str(NumNeu)+figNaStr+'.png')
    # plt.show()








