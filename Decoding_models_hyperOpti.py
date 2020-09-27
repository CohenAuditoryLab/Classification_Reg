import numpy as np
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization

#Import metrics
from metrics import get_R2
from metrics import get_rho

#Import decoder functions
# from decoders import WienerCascadeDecoder
# from decoders import WienerFilterDecoder
# from decoders import SVRDecoder
# from decoders import DenseNNDecoder
from decoders import SimpleRNNDecoder
from decoders import LSTMDecoder
from decoders import GRUDecoder
# from decoders import XGBoostDecoder

def AllDecoders(X,y):
    train_R2=[]
    test_R2=[]
    valid_R2=[]
    y_train_pred=[]
    y_test_pred=[]
    y_valid_pred=[]
    BestParams={}
    # hyperparameter sets of each model 
    models=['RNN','LSTM','GRU']
    # decoders = [
    #     SimpleRNNDecoder(),
    #     LSTMDecoder(),
    #     GRUDecoder()]
    # params=[{'units':(50,100.99),'dropout':(0,0.2),'num_epochs':(2,5.99)}]

    params=[{'units':(50,600),'dropout':(0,0.6),'num_epochs':(2,31)}]
    initpoints=10
    niter=10
    k=10

    #split training, testing datasets, and Z score input 'X' and 'y'
    # X_train_temp, X_test, y_train_temp, y_test = train_test_split(X, y, test_size=0.2)
    # X_train, X_valid, y_train, y_valid = train_test_split(X_train_temp, y_train_temp, test_size=0.2)
    training_range=[.2,1]
    valid_range=[0,.1]
    testing_range=[.1,.2]
    num_examples=X.shape[0]
    training_set=np.arange(np.int(np.round(training_range[0]*num_examples)),np.int(np.round(training_range[1]*num_examples)))
    valid_set=np.arange(np.int(np.round(valid_range[0]*num_examples)),np.int(np.round(valid_range[1]*num_examples)))    
    testing_set=np.arange(np.int(np.round(testing_range[0]*num_examples)),np.int(np.round(testing_range[1]*num_examples)))
    #Get training data
    X_train=X[training_set,:,:]
    y_train=y[training_set,:]   
    #Get testing data
    X_test=X[testing_set,:,:]
    y_test=y[testing_set,:]
    #Get validation data
    X_valid=X[valid_set,:,:]
    y_valid=y[valid_set,:]

    X_mean=np.nanmean(X,axis=0)
    X_std=np.nanstd(X,axis=0)
    X_train=np.nan_to_num((X_train-X_mean)/X_std)
    X_valid=np.nan_to_num((X_valid-X_mean)/X_std)
    X_test=np.nan_to_num((X_test-X_mean)/X_std)  
    
    #Zero-center outputs
    y_mean=np.mean(y,axis=0)
    y_train=y_train-y_mean
    y_test=y_test-y_mean
    y_valid=y_valid-y_mean

    # # RNN DECODERS
    # def RNN_evaluate(units,dropout,num_epochs):
    #     units=int(units)
    #     num_epochs=int(num_epochs)
    #     model_RNN=SimpleRNNDecoder(units,dropout,num_epochs)
    #     model_RNN.fit(X_train,y_train)
    #     y_valid_pred=model_RNN.predict(X_valid)
    #     return np.mean(get_R2(y_valid,y_valid_pred))
    # # TUNING DECODERS
    # RNN_BO=BayesianOptimization(RNN_evaluate,params[0],verbose=0)
    # RNN_BO.maximize(init_points=initpoints, n_iter=niter, kappa=k) 
    # best_params=RNN_BO.max['params']
    # best_params['units']=int(best_params['units'])
    # best_params['num_epochs']=int(best_params['num_epochs'])
    # # predict test data
    # model_RNN=SimpleRNNDecoder(best_params['units'],best_params['dropout'],best_params['num_epochs'])
    # model_RNN.fit(X_train,y_train)
    # y_valid_pred_temp=model_RNN.predict(X_valid)
    # y_test_pred_temp=model_RNN.predict(X_test)
    # y_valid_pred.append(y_valid_pred_temp)
    # y_test_pred.append(y_test_pred_temp)
    # valid_R2.append(np.mean(get_R2(y_valid,y_valid_pred_temp)))
    # test_R2.append(np.mean(get_R2(y_test,y_test_pred_temp)))
    # BestParams[models[0]]=best_params

    # LSTM DECODERS
    def LSTM_evaluate(units,dropout,num_epochs):
        units=int(units)
        num_epochs=int(num_epochs)
        model_LSTM=LSTMDecoder(units,dropout,num_epochs)
        model_LSTM.fit(X_train,y_train)
        y_valid_pred=model_LSTM.predict(X_valid)
        return np.mean(get_R2(y_valid,y_valid_pred))
    # TUNING DECODER
    LSTM_BO=BayesianOptimization(LSTM_evaluate,params[0],verbose=1)
    LSTM_BO.maximize(init_points=initpoints, n_iter=niter, kappa=k) 
    best_params=LSTM_BO.max['params']
    best_params['units']=int(best_params['units'])
    best_params['num_epochs']=int(best_params['num_epochs'])
    # predict test data
    model_LSTM=LSTMDecoder(best_params['units'],best_params['dropout'],best_params['num_epochs'])
    model_LSTM.fit(X_train,y_train)
    y_train_pred_temp=model_LSTM.predict(X_train)
    y_valid_pred_temp=model_LSTM.predict(X_valid)
    y_test_pred_temp=model_LSTM.predict(X_test)
    y_train_pred.append(y_train_pred_temp)
    y_valid_pred.append(y_valid_pred_temp)
    y_test_pred.append(y_test_pred_temp)
    train_R2.append(np.mean(get_R2(y_train,y_train_pred_temp)))
    valid_R2.append(np.mean(get_R2(y_valid,y_valid_pred_temp)))
    test_R2.append(np.mean(get_R2(y_test,y_test_pred_temp)))
    BestParams[models[1]]=best_params

    # GRU DECODERS
    def GRU_evaluate(units,dropout,num_epochs):
        units=int(units)
        num_epochs=int(num_epochs)
        model_GRU=GRUDecoder(units,dropout,num_epochs)
        model_GRU.fit(X_train,y_train)
        y_valid_pred=model_GRU.predict(X_valid)
        return np.mean(get_R2(y_valid,y_valid_pred))
    # TUNING DECODERS
    GRU_BO=BayesianOptimization(GRU_evaluate,params[0],verbose=0)
    GRU_BO.maximize(init_points=initpoints, n_iter=niter, kappa=k) 
    best_params=GRU_BO.max['params']
    best_params['units']=int(best_params['units'])
    best_params['num_epochs']=int(best_params['num_epochs'])
    # predict test data
    model_GRU=GRUDecoder(best_params['units'],best_params['dropout'],best_params['num_epochs'])
    model_GRU.fit(X_train,y_train)
    y_train_pred_temp=model_GRU.predict(X_train)
    y_valid_pred_temp=model_GRU.predict(X_valid)
    y_test_pred_temp=model_GRU.predict(X_test)
    y_train_pred.append(y_train_pred_temp)
    y_valid_pred.append(y_valid_pred_temp)
    y_test_pred.append(y_test_pred_temp)
    train_R2.append(np.mean(get_R2(y_train,y_train_pred_temp)))
    valid_R2.append(np.mean(get_R2(y_valid,y_valid_pred_temp)))
    test_R2.append(np.mean(get_R2(y_test,y_test_pred_temp)))
    BestParams[models[2]]=best_params

    return train_R2, valid_R2, test_R2, y_train_pred, y_valid_pred, y_test_pred, y_train, y_valid, y_test, BestParams
    

