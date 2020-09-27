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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Import models 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Dropout
from keras.utils import to_categorical

# load data
folder='/Users/caihuaizhen/Desktop/Cohen Lab/Projects/Classification/InputData/'
data=io.loadmat(folder+'LFP_RT_behave_nonAC_neuronwise.mat')   

# AllX=data['New_Raster_delZero'] #Load raster of all neurons
AllX=data['New_LFP'] # load LFP
AllX=AllX[:,510:] # tease the signal before stimulus onset, spike from 500, LFP from 510
Ally=data['behave_data'] #Load behave_data as y
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
# cmXYlabels=['Hit','Miss']

# Get validation data : have the same proportion of each class as input
X_train, X_valid_temp, y_train, y_valid_temp = train_test_split(X, y, test_size=0.3,stratify=y)
X_valid, X_test,y_valid, y_test = train_test_split(X_valid_temp, y_valid_temp, test_size=0.5,stratify=y_valid_temp)

##################################### CLASSIFICATION MODELS################################
##################### CNN
# CNN expects input data looks like (samples, timesteps,features)
X_train_cnn = X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_valid_cnn = X_valid.reshape(X_valid.shape[0],X_valid.shape[1],1) 
X_test_cnn = X_test.reshape(X_test.shape[0],X_test.shape[1],1) 
 
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=300, activation='relu', input_shape=(X_train_cnn.shape[1],X_train_cnn.shape[2])))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=100, activation='relu'))
model.add(Conv1D(filters=64, kernel_size=50, activation='relu'))
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(Dropout(0.2))
# model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# fit network
history = model.fit(X_train_cnn, y_train, epochs=5, batch_size=32, validation_data=(X_valid_cnn, y_valid))
# evaluate model
y_pred=model.predict_classes(X_test_cnn)

fig1=plt.figure(1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
test_loss, test_acc = model.evaluate(X_test_cnn,  y_test, verbose=2)
print(test_acc)
fig1.savefig('Accuracy_CNN.png')

fig2=plt.figure(2)
conf_matrix=confusion_matrix(y_test,y_pred,labels=cmLabel,normalize='true')
xx, yy = np.meshgrid([0,1], [0,1]) # generate ticks for text in matshow
ax = plt.gca()
cax = ax.matshow(conf_matrix,cmap=plt.cm.Blues,vmin=0, vmax=1)
zz=conf_matrix.flatten()
for v, (x_val, y_val) in enumerate(zip(xx.flatten(),yy.flatten())):
    ax.text(x_val-0.1,y_val,round(zz[v],2),color='r',fontweight='bold',va='center', ha='center')
plt.title('CM_CNN')   
fig2.colorbar(cax,ax=ax)
ax.set_xticklabels([''] + cmXYlabels)
ax.set_yticklabels([''] + cmXYlabels)
plt.xlabel('Predicted')
plt.ylabel('True') 
fig1.suptitle('Confusion matrix of CNN')
fig2.savefig('ConfMat_CNN.png')
# plt.show()
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))