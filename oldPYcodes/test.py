import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()





#Import standard packages
import numpy as np
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
from Neural_Decoding.preprocessing_funcs import get_spikes_with_history

# load data
folder='/Users/caihuaizhen/Desktop/Cohen Lab/Projects/Neural_Decoding-master/Decoding_Data/LFP/'
data=io.loadmat(folder+'LFP_RT_behave_AC.mat')   
AllX=data['LFPBefTar'] #Load raster of all neurons
Ally=data['behave_data'] #Load y
Ally=np.ravel(Ally)

# balance number of samples across classes
clss,clssnum=np.unique(Ally,return_counts=True)
ind1=np.transpose(np.array(np.where(Ally==clss[0])))
ind2=np.ravel(np.transpose(np.array(np.where(Ally==clss[1]))))
ind1=np.ravel((np.random.choice(np.squeeze(ind1),len(ind2),replace=False)).reshape(len(ind2),1))
X=np.concatenate((np.take(AllX,ind1,axis=0),np.take(AllX,ind2,axis=0)),axis=0)
y=np.concatenate((np.take(Ally,ind1,axis=0),np.take(Ally,ind2,axis=0)),axis=0)

#Set what part of data should be part of the training/testing/validation sets
num_folds=5

# initialize the array to save accuracy for all cross validation folds 
accuracy_train_logist=np.empty((num_folds,1))
accuracy_valid_logist=np.empty((num_folds,1))
accuracy_train_knn=np.empty((num_folds,1))
accuracy_valid_knn=np.empty((num_folds,1))
accuracy_train_SVM=np.empty((num_folds,1))
accuracy_valid_SVM=np.empty((num_folds,1))
accuracy_train_NB=np.empty((num_folds,1))
accuracy_valid_NB=np.empty((num_folds,1))
accuracy_train_RF=np.empty((num_folds,1))
accuracy_valid_RF=np.empty((num_folds,1))

#Hold testing data for testing after hyperparameter optimization
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2)
X_flat_train_temp=X_train_temp.reshape(X_train_temp.shape[0],(X_train_temp.shape[1]*X_train_temp.shape[2]))
X_flat_test=X_test.reshape(X_test.shape[0],(X_test.shape[1]*X_test.shape[2]))

# normalize input for Naive Bayes: scale features to [0,1]
min_max_scaler = MinMaxScaler()
X_flat_train_temp=min_max_scaler.fit_transform(X_flat_train_temp)

# split data for cross validation 
KF=KFold(n_splits=num_folds,shuffle=False)
i=-1

# training and testing models
for train_index, test_index in KF.split(X_train_temp):
    i+=1
    #Get training data
    X_train=np.take(X_train_temp,train_index,axis=0)
    y_train=np.take(y_train_temp,train_index,axis=0)
    #Get validation data
    X_valid=np.take(X_train_temp,test_index,axis=0)
    y_valid=np.take(y_train_temp,test_index,axis=0)
    
    # for imbalanced dataset, calculate the null accuracy according to testing class distribution
    yIndex=pd.Index(y_valid)
    print('Percent of each class in the validation set:(null accuracy is the largest %)')
    print(yIndex.value_counts(normalize=True,ascending=True))

    # get flattened input X for training and validation
    X_flat_train=np.take(X_flat_train_temp,train_index,axis=0)
    X_flat_valid=np.take(X_flat_train_temp,test_index,axis=0)

##################################### CLASSIFICATION MODELS################################
#####################LogisticRegression
    model_name = "Logistic Regression Classifier"
    LogisticRegressionClassifier = LogisticRegression(penalty='l2',C=1.0, class_weight='balanced',solver='lbfgs',max_iter=1000,multi_class='auto')
    LogisticRegressionClassifier.fit(X_flat_train,y_train)
    # predict accuracy 
    accuracy_train_logist[i]=LogisticRegressionClassifier.score(X_flat_train,y_train)
    accuracy_valid_logist[i]=LogisticRegressionClassifier.score(X_flat_valid,y_valid)
    # plot PR curve
    lab='Fold %d' % (i+1)
    fig1=plt.figure(1)
    ax1=plt.gca()
    disp=plot_precision_recall_curve(knnClassifier,X_flat_valid,y_valid,ax=ax1) 
    # plot confusion matrix
    fig2=plt.figure(2)
    fig2.suptitle('Confusion matrix of the LogRegClassifier')
    ax2=fig2.add_subplot(2,3,i+1)
    disp=plot_confusion_matrix(LogisticRegressionClassifier,X_flat_valid,y_valid,display_labels=['Hit','Miss'],normalize='true',ax=ax2)
    ax2.set_title('Fold%d' %(i+1))
ax1.set_title('PR curve of LogRegClassifier')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')    
plt.show()
print('TrainAccuracy:',accuracy_train_logist)
print('ValidAccuracy',accuracy_valid_logist)

#####################KNN
    model_name = 'K-Nearest Neighbor Classifier'
    knnClassifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2) #euclidean distance metric was used for the tree
    knnClassifier.fit(X_flat_train,y_train)
    # predict accuracy 
    accuracy_train_knn[i]=knnClassifier.score(X_flat_train,y_train)
    accuracy_valid_knn[i]=knnClassifier.score(X_flat_valid,y_valid)   
    # plot PR curve
    lab='Fold %d' % (i+1)
    fig3=plt.figure(3)
    ax3=plt.gca()
    disp=plot_precision_recall_curve(knnClassifier,X_flat_valid,y_valid,ax=ax3) 
    # plot confusion matrix
    fig4=plt.figure(4)
    fig4.suptitle('Confusion matrix of the KNNClassifier')
    ax4=fig4.add_subplot(2,3,i+1)
    disp=plot_confusion_matrix(knnClassifier,X_flat_valid,y_valid,display_labels=['Hit','Miss'],normalize='true',ax=ax4)
    ax4.set_title('Fold%d' %(i+1))
ax3.set_title('PR curve of KNNClassifier')
plt.show()
print('TrainAccuracy:',accuracy_train_knn)
print('ValidAccuracy',accuracy_valid_knn)

#####################SVM MODEL
    model_name='Kernel SVM Classifier'
    svmClassifier=SVC(C=0.5,kernel='rbf',gamma='auto',class_weight='balanced')
    svmClassifier.fit(X_flat_train,y_train)
    # predict accuracy 
    accuracy_train_SVM[i]=svmClassifier.score(X_flat_train,y_train)
    accuracy_valid_SVM[i]=svmClassifier.score(X_flat_valid,y_valid)
    # plot PR curve
    y_score=svmClassifier.decision_function(X_flat_valid)
    average_precision=average_precision_score(y_valid,y_score)# rectangular approximated area below the PR curve, larger, better  
    prec, recall, _ =precision_recall_curve(y_valid, y_score) # estimate precision&recall    
    lab='Fold %d AP=%.4F' % (i+1,average_precision)
    fig1=plt.figure(1)
    ax1=plt.gca()
    plt.step(recall,prec,label=lab)
    # plot confusion matrix
    fig=plt.figure(2)
    fig.suptitle('Confusion matrix of the SVMclassifier in cross validation')
    ax2=fig.add_subplot(2,3,i+1)
    disp=plot_confusion_matrix(svmClassifier,X_flat_valid,y_valid,display_labels=['Hit','Miss'],normalize='true',ax=ax2)
    ax2.set_title('Fold%d' %(i+1))
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')    
plt.show()
print('TrainAccuracy:',accuracy_train_SVM)
print('ValidAccuracy',accuracy_valid_SVM)

#####################Naive Bayes MODEL
    model_name='Naive Bayes Classifier'
    NBClassifier=MultinomialNB(alpha=0.2)
    NBClassifier.fit(X_flat_train,y_train)
    # predict accuracy 
    accuracy_train_NB[i]=NBClassifier.score(X_flat_train,y_train)
    accuracy_valid_NB[i]=NBClassifier.score(X_flat_valid,y_valid)
    # plot PR curve
    lab='Fold %d' % (i+1)
    fig1=plt.figure(1)
    ax1=plt.gca()
    disp=plot_precision_recall_curve(NBClassifier,X_flat_valid,y_valid,ax=ax1)  
    # plot confusion matrix
    fig=plt.figure(2)
    fig.suptitle('Confusion matrix of the SVMclassifier in cross validation')
    ax2=fig.add_subplot(2,3,i+1)
    disp=plot_confusion_matrix(NBClassifier,X_flat_valid,y_valid,display_labels=['Hit','Miss'],normalize='true',ax=ax2)
    ax2.set_title('Fold%d' %(i+1))
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')    
plt.show()
print('TrainAccuracy:',accuracy_train_NB)
print('ValidAccuracy',accuracy_valid_NB)

######################random forest
    model_name='Random Forest Classifier'
    RanForeClassifier=RandomForestClassifier(n_estimators=100,max_depth=2,min_samples_split=0.05,max_features='sqrt',class_weight='balanced')
    RanForeClassifier.fit(X_flat_train,y_train)
    # get feature importance of the input
    FtIm=RanForeClassifier.feature_importances_
    feature_importances=FtIm.reshape((X_test.shape[1],X_test.shape[2]))

    # predict accuracy 
    accuracy_train_RF[i]=RanForeClassifier.score(X_flat_train,y_train)
    accuracy_valid_RF[i]=RanForeClassifier.score(X_flat_valid,y_valid)
    # plot PR curve
    lab='Fold %d' % (i+1)
    fig1=plt.figure(1)
    ax1=plt.gca()
    disp=plot_precision_recall_curve(RanForeClassifier,X_flat_valid,y_valid,ax=ax1)  
    # plot confusion matrix
    fig=plt.figure(2)
    fig.suptitle('Confusion matrix of the RanForestclassifier in cross validation')
    ax2=fig.add_subplot(2,3,i+1)
    disp=plot_confusion_matrix(RanForeClassifier,X_flat_valid,y_valid,display_labels=['Hit','Miss'],normalize='true',ax=ax2)
    ax2.set_title('Fold%d' %(i+1))
    # plot feature importance matrix of the input
    fig3=plt.figure(3)
    ax0=fig3.add_subplot(2,3,i+1)
    time=np.arange(feature_importances.shape[1])
    kk=0
    for y in feature_importances:
        kk=kk+1
        lab='Channel %d' % (kk)
        plt.plot(time,y,label=lab)    
    ax0.set_xlabel('Time (ms)')
    plt.legend(fontsize=4) 

ax0.set_title('Feature importance')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')    
plt.show()
print('TrainAccuracy:',accuracy_train_RF)
print('ValidAccuracy',accuracy_valid_RF)

######################






#Import standard packages
import numpy as np
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
from Neural_Decoding.preprocessing_funcs import get_spikes_with_history

#Import metrics
from Neural_Decoding.metrics import get_R2
from Neural_Decoding.metrics import get_rho

# load data
folder='/Users/caihuaizhen/Desktop/Cohen Lab/Projects/Neural_Decoding-master/Decoding_Data/LFP/'
data=io.loadmat(folder+'LFP_RT_behave_AC.mat')   
AllX=data['LFPBefTar'] #Load raster of all neurons
Ally=data['behave_data'] #Load y
Ally=np.ravel(Ally)

# balance number of samples across classes
clss,clssnum=np.unique(Ally,return_counts=True)
ind1=np.transpose(np.array(np.where(Ally==clss[0])))
ind2=np.ravel(np.transpose(np.array(np.where(Ally==clss[1]))))
ind1=np.ravel((np.random.choice(np.squeeze(ind1),len(ind2),replace=False)).reshape(len(ind2),1))
X=np.concatenate((np.take(AllX,ind1,axis=0),np.take(AllX,ind2,axis=0)),axis=0)
y=np.concatenate((np.take(Ally,ind1,axis=0),np.take(Ally,ind2,axis=0)),axis=0)

#Set what part of data should be part of the training/testing/validation sets
num_folds=5

accuracy_train_logist=np.empty((num_folds,1))
accuracy_valid_logist=np.empty((num_folds,1))
accuracy_train_knn=np.empty((num_folds,1))
accuracy_valid_knn=np.empty((num_folds,1))
accuracy_train_SVM=np.empty((num_folds,1))
accuracy_valid_SVM=np.empty((num_folds,1))
accuracy_train_NB=np.empty((num_folds,1))
accuracy_valid_NB=np.empty((num_folds,1))
accuracy_train_RF=np.empty((num_folds,1))
accuracy_valid_RF=np.empty((num_folds,1))

#Hold testing data
X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2)
X_flat_train_temp=X_train_temp.reshape(X_train_temp.shape[0],(X_train_temp.shape[1]*X_train_temp.shape[2]))
X_flat_test=X_test.reshape(X_test.shape[0],(X_test.shape[1]*X_test.shape[2]))

# normalize input for Naive Bayes: scale features to [0,1]
min_max_scaler = MinMaxScaler()
X_flat_train_temp=min_max_scaler.fit_transform(X_flat_train_temp)

# split data for cross validation 
KF=KFold(n_splits=num_folds,shuffle=False)
i=-1

for train_index, test_index in KF.split(X_train_temp):
    i+=1
    #Get training data
    X_train=np.take(X_train_temp,train_index,axis=0)
    y_train=np.take(y_train_temp,train_index,axis=0)
    #Get validation data
    X_valid=np.take(X_train_temp,test_index,axis=0)
    y_valid=np.take(y_train_temp,test_index,axis=0)
    
    # for imbalanced dataset, calculate the null accuracy according to testing class distribution
    yIndex=pd.Index(y_valid)
    print('Percent of each class in the validation set:(null accuracy is the largest %)')
    print(yIndex.value_counts(normalize=True,ascending=True))

    X_flat_train=np.take(X_flat_train_temp,train_index,axis=0)
    X_flat_valid=np.take(X_flat_train_temp,test_index,axis=0)

##################################### CLASSIFICATION MODELS################################
######################LogisticRegression
#     model_name = "Logistic Regression Classifier"
#     LogisticRegressionClassifier = LogisticRegression(penalty='l2',C=1.0, class_weight='balanced',solver='lbfgs',max_iter=1000,multi_class='auto')
#     LogisticRegressionClassifier.fit(X_flat_train,y_train)
#     # predict accuracy 
#     accuracy_train_logist[i]=LogisticRegressionClassifier.score(X_flat_train,y_train)
#     accuracy_valid_logist[i]=LogisticRegressionClassifier.score(X_flat_valid,y_valid)
#     # plot PR curve
#     y_score=LogisticRegressionClassifier.decision_function(X_flat_valid)
#     average_precision=average_precision_score(y_valid,y_score)# rectangular approximated area below the PR curve, larger, better  
#     prec, recall, _ =precision_recall_curve(y_valid, y_score) # estimate precision&recall    
#     lab='Fold %d AP=%.4F' % (i+1,average_precision)
#     fig1=plt.figure(1)
#     ax1=plt.gca()
#     plt.step(recall,prec,label=lab)
#     # plot confusion matrix
#     fig=plt.figure(2)
#     fig.suptitle('Confusion matrix of the LogRegclassifier in cross validation')
#     ax2=fig.add_subplot(2,3,i+1)
#     disp=plot_confusion_matrix(LogisticRegressionClassifier,X_flat_valid,y_valid,display_labels=['Hit','Miss'],normalize='true',ax=ax2)
#     ax2.set_title('Fold%d' %(i+1))
# ax1.set_xlabel('Recall')
# ax1.set_ylabel('Precision')    
# plt.show()
# print('TrainAccuracy:',accuracy_train_logist)
# print('ValidAccuracy',accuracy_valid_logist)

######################KNN
#     model_name = 'K-Nearest Neighbor Classifier'
#     knnClassifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2) #euclidean distance metric was used for the tree
#     knnClassifier.fit(X_flat_train,y_train)
#     y_predict=knnClassifier.predict(X_flat_valid)
#     # predict accuracy 
#     accuracy_train_knn[i]=knnClassifier.score(X_flat_train,y_train)
#     accuracy_valid_knn[i]=knnClassifier.score(X_flat_valid,y_valid)
    
#     # plot PR curve
#     # plot confusion matrix
#     fig=plt.figure(2)
#     fig.suptitle('Confusion matrix of the KNclassifier in cross validation')
#     ax2=fig.add_subplot(2,3,i+1)
#     disp=plot_confusion_matrix(knnClassifier,X_flat_valid,y_valid,display_labels=['Hit','Miss'],normalize='true',ax=ax2)
#     ax2.set_title('Fold%d' %(i+1))
# plt.show()
# print('TrainAccuracy:',accuracy_train_knn)
# print('ValidAccuracy',accuracy_valid_knn)

######################SVM MODEL
#     model_name='Kernel SVM Classifier'
#     svmClassifier=SVC(C=0.5,kernel='rbf',gamma='auto',class_weight='balanced')
#     svmClassifier.fit(X_flat_train,y_train)
#     # predict accuracy 
#     accuracy_train_SVM[i]=svmClassifier.score(X_flat_train,y_train)
#     accuracy_valid_SVM[i]=svmClassifier.score(X_flat_valid,y_valid)
#     # plot PR curve
#     y_score=svmClassifier.decision_function(X_flat_valid)
#     average_precision=average_precision_score(y_valid,y_score)# rectangular approximated area below the PR curve, larger, better  
#     prec, recall, _ =precision_recall_curve(y_valid, y_score) # estimate precision&recall    
#     lab='Fold %d AP=%.4F' % (i+1,average_precision)
#     fig1=plt.figure(1)
#     ax1=plt.gca()
#     plt.step(recall,prec,label=lab)
#     # plot confusion matrix
#     fig=plt.figure(2)
#     fig.suptitle('Confusion matrix of the SVMclassifier in cross validation')
#     ax2=fig.add_subplot(2,3,i+1)
#     disp=plot_confusion_matrix(svmClassifier,X_flat_valid,y_valid,display_labels=['Hit','Miss'],normalize='true',ax=ax2)
#     ax2.set_title('Fold%d' %(i+1))
# ax1.set_xlabel('Recall')
# ax1.set_ylabel('Precision')    
# plt.show()
# print('TrainAccuracy:',accuracy_train_SVM)
# print('ValidAccuracy',accuracy_valid_SVM)

######################Naive Bayes MODEL
#     model_name='Naive Bayes Classifier'
#     NBClassifier=MultinomialNB(alpha=0.2)
#     NBClassifier.fit(X_flat_train,y_train)
#     # predict accuracy 
#     accuracy_train_NB[i]=NBClassifier.score(X_flat_train,y_train)
#     accuracy_valid_NB[i]=NBClassifier.score(X_flat_valid,y_valid)
#     # plot PR curve
#     lab='Fold %d' % (i+1)
#     fig1=plt.figure(1)
#     ax1=plt.gca()
#     disp=plot_precision_recall_curve(NBClassifier,X_flat_valid,y_valid,ax=ax1)  
#     # plot confusion matrix
#     fig=plt.figure(2)
#     fig.suptitle('Confusion matrix of the SVMclassifier in cross validation')
#     ax2=fig.add_subplot(2,3,i+1)
#     disp=plot_confusion_matrix(NBClassifier,X_flat_valid,y_valid,display_labels=['Hit','Miss'],normalize='true',ax=ax2)
#     ax2.set_title('Fold%d' %(i+1))
# ax1.set_xlabel('Recall')
# ax1.set_ylabel('Precision')    
# plt.show()
# print('TrainAccuracy:',accuracy_train_NB)
# print('ValidAccuracy',accuracy_valid_NB)

######################random forest
    model_name='Random Forest Classifier'
    RanForeClassifier=RandomForestClassifier(n_estimators=100,max_depth=2,min_samples_split=0.05,max_features='sqrt',class_weight='balanced')
    RanForeClassifier.fit(X_flat_train,y_train)
    # get feature importance of the input
    FtIm=RanForeClassifier.feature_importances_
    feature_importances=FtIm.reshape((X_test.shape[1],X_test.shape[2]))

    # predict accuracy 
    accuracy_train_RF[i]=RanForeClassifier.score(X_flat_train,y_train)
    accuracy_valid_RF[i]=RanForeClassifier.score(X_flat_valid,y_valid)
    # plot PR curve
    lab='Fold %d' % (i+1)
    fig1=plt.figure(1)
    ax1=plt.gca()
    disp=plot_precision_recall_curve(RanForeClassifier,X_flat_valid,y_valid,ax=ax1)  
    # plot confusion matrix
    fig=plt.figure(2)
    fig.suptitle('Confusion matrix of the RanForestclassifier in cross validation')
    ax2=fig.add_subplot(2,3,i+1)
    disp=plot_confusion_matrix(RanForeClassifier,X_flat_valid,y_valid,display_labels=['Hit','Miss'],normalize='true',ax=ax2)
    ax2.set_title('Fold%d' %(i+1))
    # plot feature importance matrix of the input
    fig3=plt.figure(3)
    ax0=fig3.add_subplot(2,3,i+1)
    time=np.arange(feature_importances.shape[1])
    kk=0
    for y in feature_importances:
        kk=kk+1
        lab='Channel %d' % (kk)
        plt.plot(time,y,label=lab)    
    ax0.set_xlabel('Time (ms)')
    plt.legend(fontsize=4) 

ax0.set_title('Feature importance')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')    
plt.show()
print('TrainAccuracy:',accuracy_train_RF)
print('ValidAccuracy',accuracy_valid_RF)

######################
