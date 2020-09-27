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
from sklearn.model_selection import GridSearchCV

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

posiLabel=1#the label of the positive class
cmXYlabels=['0','1']# confusion matrix plot xyticklabels
cmLabel=[0,posiLabel]# confusion matrix function labels

# # load models
folds=5
models=['LoRe','KNN','svm','NB','RF']
classifiers = [
     LogisticRegression(class_weight='balanced',max_iter=2000),
     KNeighborsClassifier(leaf_size=5),
     SVC(class_weight='balanced'),
     MultinomialNB(),
     RandomForestClassifier(class_weight='balanced')]
param_grid=[{'penalty':['l1','l2'], 'C':np.logspace(-6, 0, 10)},
            {'n_neighbors':range(1, 21, 2),'metric':['euclidean', 'manhattan', 'minkowski']},
            {'C':np.logspace(-6, 0, 10),'kernel':['poly', 'rbf', 'sigmoid']},
            {'alpha':np.logspace(-6, 0, 10)},
            {'n_estimators':[10, 100, 1000],'min_samples_split':range(2,10,2),'max_features':['sqrt', 'log2']} ]

# generate fake data
import random
sam=2500
mu0=0
sigma0=10
X0=[]
for i in range(sam):  
    temp = random.gauss(mu0, sigma0)  
    X0.append(temp) 
X0=np.array(X0)
X0=X0.reshape((sam,1))
y0=np.zeros((sam,1))

mu1=[5,20,100]
sigma1=[10,10,10]

Allconf_matrix_list=[]
Allaccuracy_train=np.empty([len(mu1),len(models)])
Allaccuracy_test=np.empty([len(mu1),len(models)])
Allaverage_precision=np.empty([len(mu1),len(models)])

# fig1 = plt.figure(1,figsize=(18,10))
# cm = plt.cm.RdBu
# cm_bright = ListedColormap(['#FF0000', '#0000FF'])
kk=1
for i in range(len(mu1)):
    # # generate data for cls 2
    X1=[]
    for j in range(sam):  
        temp = random.gauss(mu1[i], sigma1[i])  
        X1.append(temp) 
    X1=np.array(X1)
    X1=X1.reshape((sam,1))
    y1=np.ones((sam,1))
    X=np.concatenate((X0,X1),axis=0)
    y=np.concatenate((y0,y1),axis=0) 
    y=np.ravel(y)

    # # initialize the arrays to save accuracy for each iteration in all models
    conf_matrix_list=[]
    accuracy_train=np.empty((len(models)))
    accuracy_test=np.empty((len(models)))
    average_precision=np.empty((len(models)))
    # normalize input for Naive Bayes: scale features to [0,1]
    X_norm=MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2,stratify=y)

    # x_min, x_max = X_norm[:, 0].min() - .5, X_norm[:, 0].max() + .5
    # y_min, y_max = X_norm[:, 1].min() - .5, X_norm[:, 1].max() + .5
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    # j=0

    k=0
    # iterate over classifiers
    for param, clf in zip(param_grid, classifiers):
        # hyperparameter tuning
        grid_search = GridSearchCV(estimator = clf, param_grid = param, cv = folds, n_jobs = -1, verbose = 0)
        grid_search.fit(X_train, y_train)        
        score = grid_search.score(X_test, y_test)

        # # Plot the decision boundary. For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].
        # ax1 = fig1.add_subplot(len(mu1), len(classifiers), kk)
        # kk+=1
        # if hasattr(clf, "decision_function"):
        #     Z = grid_search.decision_function(np.c_[xx.ravel(), yy.ravel()])
        # else:
        #     Z = grid_search.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        # # Put the result into a color plot
        # Z = Z.reshape(xx.shape)
        # ax1.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        # # Plot the training points
        # ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,alpha=0.4,edgecolors='k')
        # # Plot the testing points
        # ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,edgecolors='k')
        # ax1.set_xlim(xx.min(), xx.max())
        # ax1.set_ylim(yy.min(), yy.max())
        # ax1.set_xticks(())
        # ax1.set_yticks(()) 
        # if i==0:
        #     ax1.set_title(models[j])
        #     j+=1
        # ax1.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
        #         size=15, horizontalalignment='right')

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
        k+=1 
        
    Allconf_matrix_list.append(np.array(conf_matrix_list))
    Allaccuracy_train[i,:]= accuracy_train
    Allaccuracy_test[i,:]= accuracy_test
    Allaverage_precision[i,:]= average_precision    

    # end_time = time.monotonic()
    # print(timedelta(seconds=end_time - start_time))
    # plt.tight_layout()
    # plt.show()
    # fig1.savefig('AllClfs_fakeData.png')

fig1=plt.figure(1)
ax1=plt.gca()
x=np.arange(len(models))+1
for i in range(len(mu1)):
    label1='Train-Gaussian Cls2: Me'+str(mu1[i])+' Sig'+str(sigma1[i])
    label2='Test-Gaussian Cls2: Me'+str(mu1[i])+' Sig'+str(sigma1[i])   
    marker=['o','v','s']
    plt.plot(x,Allaccuracy_train[i,:],figure=fig1,color='b',marker=marker[i],linestyle=None,label=label1)
    plt.plot(x,Allaccuracy_test[i,:],figure=fig1,color='r',marker=marker[i],linestyle=None,label=label2)
fig1.suptitle('cross validation folds = '+str(folds)+' Gaussian Cls1: Me'+str(mu0)+' Sig'+str(sigma0))
plt.ylim(0,1.1)
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.xticks(x,models)
fig1.savefig('AllAccuracy_fakeData_FeDim-'+str(X.shape[1])+'.png')
# # plt.show()


# plot the average_precision  
fig2=plt.figure(2)
ax2=plt.gca()
for i in range(len(mu1)):
    label3=['Gaussian Cls2: Me'+str(mu1[i])+' Sig'+str(sigma1[i])]
    plt.plot(x,Allaverage_precision[i,:],figure=fig2,marker=marker[i],linestyle=None,label=label3)
plt.ylim(0,1.1)
plt.ylabel('Ave precision score')
plt.xticks(x,models)
plt.legend(loc='best')
fig2.suptitle('cross validation folds = '+str(folds)+' Gaussian Cls1: Me'+str(mu0)+' Sig'+str(sigma0))
fig2.savefig('API_All_fakeData_FeDim-'+str(X.shape[1])+'.png')
# # plt.show()

# plot the confusion matrix  

xx, yy = np.meshgrid([0,1], [0,1]) # generate ticks for text in matshow

for i in range(len(mu1)):
    fig3 = plt.figure(3,figsize=(18,10))
    for u in range(len(models)):
        conf_matrix_temp=np.squeeze(np.array(Allconf_matrix_list)[i,:,:,:])[u]
        ax = fig3.add_subplot(2,3,u+1)
        cax = ax.matshow(conf_matrix_temp,cmap=plt.cm.Blues,vmin=0, vmax=1)
        zz=conf_matrix_temp.flatten()
        for v, (x_val, y_val) in enumerate(zip(xx.flatten(),yy.flatten())):
            ax.text(x_val-0.1,y_val,round(zz[v],2),color='r',fontweight='bold',va='center', ha='center')
            # ax.text(x_val-0.1,y_val+0.1,'+/-',color='r',fontweight='bold',va='center', ha='center')
            # ax.text(x_val-0.1,y_val+0.2,round(conf_matrix_temp_sem.flatten()[v],2),color='r',fontweight='bold',va='center', ha='center')
        plt.title(models[u])   
        fig3.colorbar(cax,ax=ax)
        ax.set_xticklabels([''] + cmXYlabels)
        ax.set_yticklabels([''] + cmXYlabels)
        plt.xlabel('Predicted')
        plt.ylabel('True') 
    fig3.suptitle('Confusion matrix of all Clfs--'+'Gaussian Cls1: Me'+str(mu0)+' Sig'+str(sigma0)+' Cls2: Me'+str(mu1[i])+' Sig'+str(sigma1[i]))
    fig3.savefig('ConfMat_fakeData_FeDim-'+str(X.shape[1])+'_M_'+str(mu1[i])+' Sig_'+str(sigma1[i])+'.png')
    plt.clf()
# plt.show()







