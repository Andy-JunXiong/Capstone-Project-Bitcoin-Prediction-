import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.preprocessing import MinMaxScaler 
from time import time
from math import sqrt
from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, Ridge, BayesianRidge
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

# Data normalization
def normalize_data(X,Y=None):
    scaler = MinMaxScaler()
    return scaler.fit_transform(X,Y)

# Define time-series cross validation split
def time_series_CV_split(n_samples, test_size, step_size):
    data_split = []
    train_end = n_samples - test_size - step_size 
    test_start = train_end + step_size
    for i in range(0,test_size):
        train_index = list(range(0, train_end))
        test_index = [test_start]
        data_split.append([train_index,test_index])
        train_end+=1
        test_start+=1
    return data_split

def wrapper_feature_selector(X,Y,model,subset=np.arange(0,35).tolist()):
    sel = subset.copy() # # Selected features
    overall_error = train_and_predict(X[:,sel],Y,model,n_validation=1,predict=False,return_y=False) 
    while len(sel) != 0:
        # Select candidate
        cand_error = 1e10 # Assign a big number
        for cand in sel:
            features = sel.copy()
            features.remove(cand)
            if len(features) > 1:
                new_error = train_and_predict(X[:,features],Y,model,n_validation=1,predict=False,return_y=False)
            else:
                new_error = train_and_predict(X[:,features[0]].reshape(-1,1),Y,model,n_validation=1,predict=False,return_y=False)
            if new_error < cand_error:
                selected_candidate = cand
                cand_error = new_error
        if overall_error < cand_error:
            # Stop if the new candidate doesn’t
            # improve the assessment of the
            # previously selected candidates
            break
        else:
            overall_error = cand_error
            sel.remove(selected_candidate)
    rmse = train_and_predict(X[:,sel],Y,model,n_validation=1,predict=False,return_y=False)
    return [sel, rmse]

def train_and_predict(X,Y,model,n_validation=91,subset=np.arange(0,35).tolist(),predict=True,return_y=True):
    data_split = time_series_CV_split(len(X),n_validation,0)
    Y_pred = []
    Y_test = []
    n = 1
    for fold in data_split:
        X_train = X[fold[0]]
        Y_train = Y[fold[0]]
        X_test = X[fold[1]]
        y_pred = Y[fold[1]]

        if predict: # for prediction
            # Feature Selection
            selected_features, rmse = wrapper_feature_selector(X_train,Y_train,model,subset)         
            #print('Test Sample {} - RMSE: {:0.2f}'.format(n, rmse) + ', Selected Features:', str(selected_features))
            # Normalize Data
            scaler = MinMaxScaler()
            scaler.fit(X_train[:,selected_features],Y_train)
            X_train = scaler.transform(X_train[:,selected_features])
            X_test = scaler.transform(X_test[:,selected_features])  
        if not predict: # for training
            # Normalize Data
            scaler = MinMaxScaler()
            scaler.fit(X_train,Y_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test) 
        

        model.fit(X_train,Y_train.reshape(-1,))
        prediction = model.predict(X_test).reshape(-1,1)
        
        if predict:
            rmse = np.sqrt(mean_squared_error(prediction,y_pred))
            print('Test Sample {} - RMSE: {:0.2f}'.format(n, rmse) + ', Selected Features:', str(selected_features))
        
        
        Y_pred.append(prediction)
        Y_test.append(y_pred)
        n +=1
    Y_pred = np.asarray(Y_pred)
    Y_test = np.asarray(Y_test)
    rmse = np.sqrt(mean_squared_error(Y_pred.reshape(-1,1), Y_test.reshape(-1,1)))
    if return_y:
        return rmse, Y_test, Y_pred
    if not return_y:
        return rmse

def fine_tune_alpha(X,Y,model):
    # Generate alphas to test
    alphas = np.linspace(0.01,4.15,num=415).tolist()
    alphas.extend([0.0000001,0.000001,0.00001,0.0001,0.001])
    best_rmse = 1e10
    best_alpha = 0
    for alpha in alphas:   
    # Fit to Ridge
        if model == 'Lasso': rmse = train_and_predict(X,Y,Lasso(alpha=alpha),predict=False,return_y=False)
        elif model == 'Ridge': rmse = train_and_predict(X,Y,Ridge(alpha=alpha),predict=False,return_y=False)
        if rmse < best_rmse:
            best_alpha = alpha
            best_rmse = rmse
    return best_alpha

def fine_tune_enet(X,Y):
    # Generate alphas, l1_ratio to test
    l1_ratio = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    alphas = np.linspace(0.01,20.15,num=2014).tolist()
    alphas.extend([0.0000001,0.000001,0.00001,0.0001,0.001])
    best_rmse = 1e10
    best_l1_ratio = 0
    best_alpha = 0
    for ratio in l1_ratio: 
        for alpha in alphas:
            # Fit to Elastic Net
            rmse = train_and_predict(X,Y,ElasticNet(l1_ratio=ratio,alpha=alpha),predict=False,return_y=False)
            if rmse < best_rmse:
                best_alpha = alpha
                best_l1_ratio = ratio
                best_rmse = rmse
    return best_alpha, best_l1_ratio

def fine_tune_KNN(X,Y,subset=np.arange(0,35).tolist()):
    best_rmse = 1e10
    best_k = -1
    best_p = -1
    best_weights = 'uniform'
    for k in [5,6,7,8,9,10]:
        for weights in ['distance','uniform']:
            for p in [1,2,3]:
                rmse = train_model(X[:,subset],Y,KNeighborsRegressor(n_neighbors=k,p=p,weights=weights),predict=False)
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_k = k
                    best_p = p
                    best_weights = weights       
    return best_k,best_p,best_weights

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
