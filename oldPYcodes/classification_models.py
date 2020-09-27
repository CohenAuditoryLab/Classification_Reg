#Import standard packages
import numpy as np
import scipy
import matplotlib.pyplot as plt
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

#Import function to get the covariate matrix that includes spike history from previous bins
# from Neural_Decoding.preprocessing_funcs import get_spikes_with_history

# load data
folder='/Users/caihuaizhen/Desktop/Cohen Lab/Projects/Classification/InputData/'

data=io.loadmat(folder+'LFP_RT_behave_nonAC_neuronwise.mat')   

# AllX=data['New_Raster'] #Load raster of all neurons
AllX=data['New_LFP'] # load LFP
AllX=AllX[:,510:] # tease the signal before stimulus onset, spike from 500, LFP from 510
Ally=data['behave_data'] #Load behave_data as y
# Ally=data['loc'] #Load brain region as y

Ally=np.ravel(Ally)

# balance number of samples across classes
clss,clssnum=np.unique(Ally,return_counts=True)
ind1=np.transpose(np.array(np.where(Ally==clss[0])))
ind2=np.ravel(np.transpose(np.array(np.where(Ally==clss[2])))) #change the class for different dataset

ind1=np.ravel((np.random.choice(np.squeeze(ind1),len(ind2),replace=False)).reshape(len(ind2),1))
X=np.concatenate((np.take(AllX,ind1,axis=0),np.take(AllX,ind2,axis=0)),axis=0)
y=np.concatenate((np.take(Ally,ind1,axis=0),np.take(Ally,ind2,axis=0)),axis=0)

# cmLabel=['AC','nonAC']
cmLabel=['Hit','FA']
# cmLabel=['Hit','Miss']

#Set what part of data should be part of the training/testing/validation sets
num_folds=5

# initialize the arrays to save accuracy for all cross validation folds in all models
accuracy_train=np.empty((num_folds,5))
accuracy_valid=np.empty((num_folds,5))

#Hold testing data for testing after hyperparameter optimization
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2)

# normalize input for Naive Bayes: scale features to [0,1]
min_max_scaler = MinMaxScaler()
X_train_temp=min_max_scaler.fit_transform(X_train_temp)

# split data for cross validation 
KF=KFold(n_splits=num_folds,shuffle=False)
i=-1

# training and testing models
for train_index, test_index in KF.split(X_train_temp):
    i+=1
    #Get training data
    X_train=np.take(X_train_temp,train_index,axis=0)
    y_train=np.take(y_train_temp,train_index,axis=0)
    print('Num of trials in training:')
    print(X_train.shape[0])
    #Get validation data
    X_valid=np.take(X_train_temp,test_index,axis=0)
    y_valid=np.take(y_train_temp,test_index,axis=0)
    print('Num of trials in validating:')
    print(X_valid.shape[0])
    # for imbalanced dataset, calculate the null accuracy according to testing class distribution
    yIndex=pd.Index(y_valid)
    print('Percent of each class in the validation set:(null accuracy is the largest %)')
    print(yIndex.value_counts(normalize=True))

##################################### CLASSIFICATION MODELS################################
#####################LogisticRegression
    model_name = "Logistic Regression Classifier"
    LogisticRegressionClassifier = LogisticRegression(penalty='l2',C=1.0, class_weight='balanced',solver='lbfgs',max_iter=1000,multi_class='auto')
    LogisticRegressionClassifier.fit(X_train,y_train)
    # predict accuracy 
    accuracy_train[i,0]=LogisticRegressionClassifier.score(X_train,y_train)
    accuracy_valid[i,0]=LogisticRegressionClassifier.score(X_valid,y_valid)
    # plot PR curve
    lab='Fold %d' % (i+1)
    fig1=plt.figure(1,figsize=(8,8))
    ax1=plt.gca()
    disp=plot_precision_recall_curve(LogisticRegressionClassifier,X_valid,y_valid,ax=ax1) 
    
    # plot confusion matrix
    fig2=plt.figure(2,figsize=(18,10))
    ax2=fig2.add_subplot(2,3,i+1)
    disp=plot_confusion_matrix(LogisticRegressionClassifier,X_valid,y_valid,display_labels=cmLabel,cmap=plt.cm.Blues,normalize='true',ax=ax2)
    # plt.cm.ScalarMappable.set_clim([0,1])  
    ax2.set_title('Fold%d' %(i+1))
 
# # # #####################KNN
    model_name = 'K-Nearest Neighbor Classifier'
    knnClassifier=KNeighborsClassifier(n_neighbors=10,metric='minkowski',leaf_size=5) #euclidean distance metric was used for the tree
    knnClassifier.fit(X_train,y_train)
    # predict accuracy 
    accuracy_train[i,1]=knnClassifier.score(X_train,y_train)
    accuracy_valid[i,1]=knnClassifier.score(X_valid,y_valid)   
    # plot PR curve
    lab='Fold %d' % (i+1)
    fig3=plt.figure(3,figsize=(8,8))
    ax3=plt.gca()
    disp=plot_precision_recall_curve(knnClassifier,X_valid,y_valid,ax=ax3) 

    # plot confusion matrix
    fig4=plt.figure(4,figsize=(18,10))
    ax4=fig4.add_subplot(2,3,i+1)
    disp=plot_confusion_matrix(knnClassifier,X_valid,y_valid,display_labels=cmLabel,cmap=plt.cm.Blues,normalize='true',ax=ax4)
    ax4.set_title('Fold%d' %(i+1))

#####################SVM MODEL
    model_name='Kernel SVM Classifier'
    svmClassifier=SVC(C=0.8,kernel='rbf',gamma='scale',class_weight='balanced')
    svmClassifier.fit(X_train,y_train)
    # predict accuracy 
    accuracy_train[i,2]=svmClassifier.score(X_train,y_train)
    accuracy_valid[i,2]=svmClassifier.score(X_valid,y_valid)
    # plot PR curve
    lab='Fold %d' % (i+1)
    fig5=plt.figure(5,figsize=(8,8))
    ax5=plt.gca()
    disp=plot_precision_recall_curve(svmClassifier,X_valid,y_valid,ax=ax5)

    # plot confusion matrix
    fig6=plt.figure(6,figsize=(18,10))
    ax6=fig6.add_subplot(2,3,i+1)
    disp=plot_confusion_matrix(svmClassifier,X_valid,y_valid,display_labels=cmLabel,cmap=plt.cm.Blues,normalize='true',ax=ax6)
    ax6.set_title('Fold%d' %(i+1))

#####################Naive Bayes MODEL
    model_name='Naive Bayes Classifier'
    NBClassifier=MultinomialNB(alpha=0.1)
    NBClassifier.fit(X_train,y_train)
    # predict accuracy 
    accuracy_train[i,3]=NBClassifier.score(X_train,y_train)
    accuracy_valid[i,3]=NBClassifier.score(X_valid,y_valid)
    # plot PR curve
    lab='Fold %d' % (i+1)
    fig7=plt.figure(7,figsize=(8,8))
    ax7=plt.gca()
    disp=plot_precision_recall_curve(NBClassifier,X_valid,y_valid,ax=ax7) 

    # plot confusion matrix
    fig8=plt.figure(8,figsize=(18,10))
    ax8=fig8.add_subplot(2,3,i+1)
    disp=plot_confusion_matrix(NBClassifier,X_valid,y_valid,display_labels=cmLabel,cmap=plt.cm.Blues,normalize='true',ax=ax8)
    ax8.set_title('Fold%d' %(i+1))

#####################random forest
    model_name='Random Forest Classifier'
    RanForeClassifier=RandomForestClassifier(n_estimators=100,max_depth=2,min_samples_split=0.05,max_features='sqrt',class_weight='balanced')
    RanForeClassifier.fit(X_train,y_train)
    # get feature importance of the input
    feature_importances=RanForeClassifier.feature_importances_
    # feature_importances=FtIm.reshape((X_test.shape[1],X_test.shape[2]))

    # predict accuracy 
    accuracy_train[i,4]=RanForeClassifier.score(X_train,y_train)
    accuracy_valid[i,4]=RanForeClassifier.score(X_valid,y_valid)
    # plot PR curve
    lab='Fold %d' % (i+1)
    fig9=plt.figure(9,figsize=(8,8))
    ax9=plt.gca()
    disp=plot_precision_recall_curve(RanForeClassifier,X_valid,y_valid,ax=ax9)  

    # plot confusion matrix
    fig10=plt.figure(10,figsize=(18,10))
    ax10=fig10.add_subplot(2,3,i+1)
    disp=plot_confusion_matrix(RanForeClassifier,X_valid,y_valid,display_labels=cmLabel,cmap=plt.cm.Blues,normalize='true',ax=ax10)
    ax10.set_title('Fold%d' %(i+1))

    # plot feature importance matrix of the input
    fig11=plt.figure(11,figsize=(18,10))
    ax11=fig11.add_subplot(2,3,i+1)
    time=np.arange(feature_importances.shape[0])
    plt.plot(time,feature_importances) 
    ax11.set_xlabel('Time (ms)')
    # kk=0
    # for y in feature_importances:
    #     kk=kk+1
    #     lab='Channel %d' % (kk)
    #     plt.plot(time,y,label=lab)    

fig12=plt.figure(12)
ax12=plt.gca()
x=np.arange(5)+1
train_mean=np.mean(accuracy_train,axis=0).reshape(num_folds,1)
train_sem=np.array([np.std(accuracy_train,axis=0)*np.sqrt(1./num_folds+1./(num_folds-1))]).reshape(num_folds)
plt.errorbar(x,train_mean,yerr=train_sem,figure=fig12,marker='o',linestyle=None,label='TrainAcc')
valid_mean=np.mean(accuracy_valid,axis=0).reshape(num_folds,1)
valid_sem=np.array([np.std(accuracy_valid,axis=0)*np.sqrt(1./num_folds+1./(num_folds-1))]).reshape(num_folds)
plt.errorbar(x,valid_mean,yerr=valid_sem,figure=fig12,marker='s',linestyle=None,label='ValidAcc')
plt.ylim(0,1.1)
plt.ylabel('Accuracy')
ax12.legend()
plt.xticks(x,['LR','KNN','SVM','NB','RF'])
# plt.show()

ax1.set_title('PR curve of LogRegClassifier') 
fig2.suptitle('Confusion matrix of the LogRegClassifier')
ax3.set_title('PR curve of KNNClassifier') 
fig4.suptitle('Confusion matrix of the KNNClassifier')
ax5.set_title('PR curve of svmClassifier')  
fig6.suptitle('Confusion matrix of the SVMClassifier')
ax7.set_title('PR curve of NBClassifier')  
fig8.suptitle('Confusion matrix of the NBClassifier')
ax9.set_title('PR curve of RanForestClassifier') 
fig10.suptitle('Confusion matrix of the RanForestClassifier')    
ax11.set_title('Feature importance')
ax11.legend(fontsize=4) 

fig1.savefig('LR1.png')
fig2.savefig('LR2.png')
fig3.savefig('KNN1.png')   
fig4.savefig('KNN2.png')
fig5.savefig('SVM1.png')
fig6.savefig('SVM2.png')
fig7.savefig('NB1.png')
fig8.savefig('NB2.png')
fig9.savefig('RF1.png')
fig10.savefig('RF2.png')
fig11.savefig('RF3.png')
fig12.savefig('AllAccuracy.png')


print('TrainAccuracy: ',accuracy_train)
print('ValidAccuracy: ',accuracy_valid)
