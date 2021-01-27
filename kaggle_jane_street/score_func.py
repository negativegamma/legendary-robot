import numpy as np 
import pandas as pd 



def u2_wresp(wresp,action):
    score = wresp* action 
    mask =  score >= 0 
     
    return [np.sum(score), np.sum(score[mask]), np.sum(score[~mask])]



def u(resp, weight, action):
    score = resp * weight * action 
    return [np.sum(score)]

def u2(resp, weight, action):
    score = resp * weight * action 
    mask =  score >= 0 
    
    return [np.sum(score), np.sum(score[mask]), np.sum(score[~mask])]




def u_score(resp, weight, action):
    score = resp * weight * action 
    return [np.mean(score), np.std(score)]
def comp_score(resp, weight, action):
    return [pos_score(resp, weight, action), neg_score(resp, weight, action)]

def neg_score(resp, weight, action):
    score = resp * weight * action 
    mask =  score < 0 
    
    return [np.sum(score[mask] ), np.std(score[mask] ),(sum(mask)/len(resp))]


def pos_score(resp, weight, action):
    score = resp * weight * action 
    mask =  score >= 0 

    return [np.sum(score[mask] ), np.std(score[mask] ), sum(mask)/len(resp)]

