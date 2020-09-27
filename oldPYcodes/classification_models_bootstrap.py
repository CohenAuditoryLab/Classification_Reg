import time
from datetime import timedelta
start_time = time.monotonic()
#Import standard packages
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib
from scipy import io
from scipy import stats
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

# Import models 
from sklearn.linear_model import LogisticRegression    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical

#Import function to get the covariate matrix that includes spike history from previous bins
# from Neural_Decoding.preprocessing_funcs import get_spikes_with_history

# load data
folder='./InputData/'
data=io.loadmat(folder+'Spike_RT_behave_nonAC_neuronwise.mat')   

# AllX=data['New_Raster_delZero'] #Load raster of all neurons
# AllX=data['New_LFP'] # load LFP
AllX=data['combZscore_Firerate']# load firing rate
# AllX=AllX[:,510:] # tease the signal before stimulus onset, spike from 500, LFP from 510,firing rate from 0

Ally=data['comb_behave_data'] #Load behave_data as y
# Ally=data['loc'] #Load brain region as y
Ally=np.ravel(Ally)

# balance number of samples across classes, equal to the num of samples from the smallest class
clss,clssnum=np.unique(Ally,return_counts=True)
ind1=np.transpose(np.array(np.where(Ally==clss[0])))
ind2=np.ravel(np.transpose(np.array(np.where(Ally==clss[2])))) #change the class for different dataset

ind1=np.ravel((np.random.choice(np.squeeze(ind1),len(ind2),replace=False)).reshape(len(ind2),1))
X=np.concatenate((np.take(AllX,ind1,axis=0),np.take(AllX,ind2,axis=0)),axis=0)
y=np.concatenate((np.take(Ally,ind1,axis=0),np.take(Ally,ind2,axis=0)),axis=0)

# confusion matrix function labels
cmLabel=[0,2]

# confusion matrix plot xyticklabels
# cmXYlabels=['AC','nonAC']
cmXYlabels=['Hit','FA']
#cmXYlabels=['Hit','Miss']

#the label of the positive class
posiLabel=2

#Set bootstrap iterations
n_iterations=5
 
num_models=5
models=['LoRe','KNN','svm','NB','RF']

# initialize the arrays to save accuracy for each iteration in all models
accuracy_train=np.empty((n_iterations,5))
accuracy_valid=np.empty((n_iterations,5))
average_precision=np.empty((n_iterations,5))
y_real_all=[]
y_proba_LoRe=[]
y_proba_KNN=[]
y_proba_svm=[]
y_proba_NB=[]
y_proba_RF=[]

conf_matrix_list_LoRe=[]
conf_matrix_list_KNN=[]
conf_matrix_list_svm=[]
conf_matrix_list_NB=[]
conf_matrix_list_RF=[]

Allconf_matrix=np.empty((num_models,2,2))

# normalize input for Naive Bayes: scale features to [0,1]
min_max_scaler = MinMaxScaler()
X_norm=min_max_scaler.fit_transform(X)

# training and testing models
for i in range(n_iterations):
    # Get validation data : have the same proportion of each class as input
    X_train_temp, X_valid, y_train_temp, y_valid = train_test_split(X_norm, y, test_size=0.2,stratify=y)
    #Get training data
    AllData_train = resample(np.concatenate((X_train_temp,y_train_temp.reshape(y_train_temp.shape[0],1)),axis=1),random_state=0)
    X_train = AllData_train[:,:-1]
    y_train = AllData_train[:,-1]     
    if (i==0):
        # for imbalanced dataset, calculate the null accuracy according to testing class distribution
        yIndex=pd.Index(y_valid.reshape(y_valid.shape[0]))
        print('Percent of each class in the validation set:(null accuracy is the largest %)')
        print(yIndex.value_counts(normalize=True))
        print('Num of trials in validating:')
        print(X_valid.shape[0])
        print('Num of trials in training:')
        print(X_train.shape[0])

##################################### CLASSIFICATION MODELS################################
#####################LogisticRegression
    model_name = "Logistic Regression Classifier"
    LogisticRegressionClassifier = LogisticRegression(penalty='l2',C=1.0, class_weight='balanced',solver='lbfgs',max_iter=1000,multi_class='auto')
    LogisticRegressionClassifier.fit(X_train,y_train)
    # predict accuracy 
    accuracy_train[i,0]=LogisticRegressionClassifier.score(X_train,y_train)
    accuracy_valid[i,0]=LogisticRegressionClassifier.score(X_valid,y_valid)
    # # plot PR curve
    y_score=LogisticRegressionClassifier.decision_function(X_valid)
    precision, recall, _ = precision_recall_curve(y_valid, y_score, pos_label=posiLabel)
    average_precision[i,0] = average_precision_score(y_valid, y_score, pos_label=posiLabel)
    y_real_all.append(y_valid)
    y_proba_LoRe.append(y_score)
    # # plot confusion matrix
    conf_matrix = confusion_matrix(y_valid, LogisticRegressionClassifier.predict(X_valid),labels=cmLabel,normalize='true')
    conf_matrix_list_LoRe.append(conf_matrix)
    print('LogRe') 
# # # #####################KNN
    model_name = 'K-Nearest Neighbor Classifier'
    knnClassifier=KNeighborsClassifier(n_neighbors=10,metric='minkowski',leaf_size=5) #euclidean distance metric was used for the tree
    knnClassifier.fit(X_train,y_train)
    # predict accuracy 
    accuracy_train[i,1]=knnClassifier.score(X_train,y_train)
    accuracy_valid[i,1]=knnClassifier.score(X_valid,y_valid)   
    # # plot PR curve
    y_score=knnClassifier.predict_proba(X_valid)
    precision, recall, _ = precision_recall_curve(y_valid, y_score[:,1], pos_label=posiLabel)
    average_precision[i,1] = average_precision_score(y_valid, y_score[:,1], pos_label=posiLabel)
    y_proba_KNN.append(y_score)
    # # plot confusion matrix
    conf_matrix = confusion_matrix(y_valid, knnClassifier.predict(X_valid),labels=cmLabel,normalize='true')
    conf_matrix_list_KNN.append(conf_matrix)   
    print('KNN')
#####################SVM MODEL
    model_name='Kernel SVM Classifier'
    svmClassifier=SVC(C=0.8,kernel='rbf',gamma='scale',class_weight='balanced')
    svmClassifier.fit(X_train,y_train)
    # predict accuracy 
    accuracy_train[i,2]=svmClassifier.score(X_train,y_train)
    accuracy_valid[i,2]=svmClassifier.score(X_valid,y_valid)
    # # plot PR curve
    y_score=svmClassifier.decision_function(X_valid)
    precision, recall, _ = precision_recall_curve(y_valid, y_score, pos_label=posiLabel)
    average_precision[i,2] = average_precision_score(y_valid, y_score,pos_label=posiLabel)
    y_proba_svm.append(y_score)
    # # plot confusion matrix
    conf_matrix = confusion_matrix(y_valid, svmClassifier.predict(X_valid),labels=cmLabel,normalize='true')
    conf_matrix_list_svm.append(conf_matrix)  
    print('SVM')
#####################Naive Bayes MODEL
    model_name='Naive Bayes Classifier'
    NBClassifier=MultinomialNB(alpha=0.1)
    NBClassifier.fit(X_train,y_train)
    # predict accuracy 
    accuracy_train[i,3]=NBClassifier.score(X_train,y_train)
    accuracy_valid[i,3]=NBClassifier.score(X_valid,y_valid)
    # # plot PR curve
    y_score=NBClassifier.predict_proba(X_valid)
    precision, recall, _ = precision_recall_curve(y_valid, y_score[:,1], pos_label=posiLabel)
    average_precision[i,3] = average_precision_score(y_valid, y_score[:,1], pos_label=posiLabel)
    y_proba_NB.append(y_score)
    # # plot confusion matrix
    conf_matrix = confusion_matrix(y_valid, NBClassifier.predict(X_valid),labels=cmLabel,normalize='true')
    conf_matrix_list_NB.append(conf_matrix)  
    print('NB')
#####################random forest
    model_name='Random Forest Classifier'
    RanForeClassifier=RandomForestClassifier(n_estimators=100,max_depth=2,min_samples_split=0.05,max_features='sqrt',class_weight='balanced')
    RanForeClassifier.fit(X_train,y_train)
    # get feature importance of the input
    feature_importances=RanForeClassifier.feature_importances_
    # predict accuracy 
    accuracy_train[i,4]=RanForeClassifier.score(X_train,y_train)
    accuracy_valid[i,4]=RanForeClassifier.score(X_valid,y_valid)
    # # plot PR curve 
    y_score=RanForeClassifier.predict_proba(X_valid)
    precision, recall, _ = precision_recall_curve(y_valid, y_score[:,1], pos_label=posiLabel)
    average_precision[i,4] = average_precision_score(y_valid, y_score[:,1], pos_label=posiLabel)
    y_proba_RF.append(y_score)
    # # plot confusion matrix
    conf_matrix = confusion_matrix(y_valid, RanForeClassifier.predict(X_valid),labels=cmLabel,normalize='true')
    conf_matrix_list_RF.append(conf_matrix) 
    print('randFore')


fig12=plt.figure(12)
ax12=plt.gca()
x=np.arange(5)+1
train_mean=np.mean(accuracy_train,axis=0).reshape(num_models,1)
train_sem=np.array([np.std(accuracy_train,axis=0)*np.sqrt(1./n_iterations+1./(n_iterations-1))]).reshape(num_models)
plt.errorbar(x,train_mean,yerr=train_sem,figure=fig12,marker='o',linestyle=None,label='TrainAcc',capsize=0.5)
valid_mean=np.mean(accuracy_valid,axis=0).reshape(num_models,1)
valid_sem=np.array([np.std(accuracy_valid,axis=0)*np.sqrt(1./n_iterations+1./(n_iterations-1))]).reshape(num_models)
plt.errorbar(x,valid_mean,yerr=valid_sem,figure=fig12,marker='s',linestyle=None,label='ValidAcc',capsize=0.5)
plt.ylim(0.5 if valid_mean.min()-valid_sem.max()>0.5 else 0,1.1)
plt.ylabel('Accuracy')
ax12.legend()
plt.xticks(x,models)
fig12.suptitle('n_iterations = '+str(n_iterations))
fig12.savefig('AllAccuracy.png')
# plt.show()

# plot the average_precision across bootstrap iterations 
fig2=plt.figure(2)
ax2=plt.gca()
x=np.arange(num_models)+1
average_precision_mean=np.mean(average_precision,axis=0).reshape(num_models,1)
average_precision_sem=stats.sem(average_precision,axis=0).reshape(num_models)
plt.errorbar(x,average_precision_mean,yerr=average_precision_sem,figure=fig2,marker='o',linestyle=None,uplims=True,lolims=True)
plt.ylim(0,1.1)
plt.ylabel('Ave precision score')
plt.xticks(x,models)
fig2.suptitle('n_iterations = '+str(n_iterations))
fig2.savefig('API_All.png')
# plt.show()

# plot the confusion matrix across bootstrap iterations
conf_matrix_dict={}
conf_matrix_dict['LoReMean']=np.mean(np.array(conf_matrix_list_LoRe),axis=0)  
conf_matrix_dict['KNNMean']=np.mean(np.array(conf_matrix_list_KNN),axis=0)
conf_matrix_dict['svmMean']=np.mean(np.array(conf_matrix_list_svm),axis=0)
conf_matrix_dict['NBMean']=np.mean(np.array(conf_matrix_list_NB),axis=0)
conf_matrix_dict['RFMean']=np.mean(np.array(conf_matrix_list_RF),axis=0)
conf_matrix_dict['LoReSem']=stats.sem(np.array(conf_matrix_list_LoRe),axis=0)  
conf_matrix_dict['KNNSem']=stats.sem(np.array(conf_matrix_list_KNN),axis=0)
conf_matrix_dict['svmSem']=stats.sem(np.array(conf_matrix_list_svm),axis=0)
conf_matrix_dict['NBSem']=stats.sem(np.array(conf_matrix_list_NB),axis=0)
conf_matrix_dict['RFSem']=stats.sem(np.array(conf_matrix_list_RF),axis=0)

fig1 = plt.figure(1,figsize=(18,10))
xx, yy = np.meshgrid([0,1], [0,1]) # generate ticks for text in matshow
for u in range(num_models):
    ax = fig1.add_subplot(2,3,u+1)
    cax = ax.matshow(conf_matrix_dict[models[u]+'Mean'],cmap=plt.cm.Blues,vmin=0, vmax=1)
    zz=conf_matrix_dict[models[u]+'Mean'].flatten()
    for v, (x_val, y_val) in enumerate(zip(xx.flatten(),yy.flatten())):
        ax.text(x_val-0.1,y_val,round(zz[v],2),color='r',fontweight='bold',va='center', ha='center')
        ax.text(x_val-0.1,y_val+0.1,'+/-',color='r',fontweight='bold',va='center', ha='center')
        ax.text(x_val-0.1,y_val+0.2,round(conf_matrix_dict[models[u]+'Sem'].flatten()[v],2),color='r',fontweight='bold',va='center', ha='center')
    plt.title('CM_'+models[u])   
    fig1.colorbar(cax,ax=ax)
    ax.set_xticklabels([''] + cmXYlabels)
    ax.set_yticklabels([''] + cmXYlabels)
    plt.xlabel('Predicted')
    plt.ylabel('True') 
fig1.suptitle('Confusion matrix of all Classifiers: n_iterations = '+str(n_iterations))
fig1.savefig('ConfMat.png')
# plt.show()

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))








