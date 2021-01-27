import datatable as dt
import pandas as pd


#mask_df = pd.read_csv('~/jane-street-market-prediction/params/kfolds.csv')


def ret_fold_mask(k):
    
    mask_df = pd.read_csv('~/jane-street-market-prediction/params/kfolds.csv')
    return mask_df['date_'+str(k)]


def kfold_split(data,mask,features,target):
    train = data[~mask]
    test = data[mask]

    x_train = train.loc[:, features]
    y_train = (train.loc[:, target])
    x_test = test.loc[:, features]
    y_test= (test.loc[:, target])
    
    test_wresp = test.loc[:, 'resp'] * test.loc[:, 'weight'] 
    
    
    train_wresp = train.loc[:, 'resp'] *train.loc[:, 'weight'] 
    
    return  [x_train.reset_index(drop=True), 
            y_train.reset_index(drop=True), 
            x_test.reset_index(drop=True), 
            y_test.reset_index(drop=True), 
            train_wresp.reset_index(drop=True),
            test_wresp.reset_index(drop=True)]
#x_train, y_train, x_test, y_test = kfold_split(data,mask,features,target)