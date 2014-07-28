# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 13:13:45 2014

@author: hehu
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
    s = 1.0 / (1.0 + np.exp(-1.0 * z))
    return s
    
class IterativeTrainer():
    def __init__(self, 
                 clf = LogisticRegression(penalty = "l1", C = 10**-1.5), 
                 iters = 1, 
                 relabelWeight = 10,
                 relabelThr = 0.1,
                 thrNormalize = False,
                 decay = 1.0,
                 substitute = False):
        
        self.clf = clf
        self.iters = iters
        self.relabelWeight = relabelWeight
        self.relabelThr = relabelThr
        self.thrNormalize = thrNormalize
        self.decay = decay
        self.substitute = substitute
        
    def fit(self, X, y, X_test):
        
        self.clf.fit(X, y)
        
        # Run relabeling iterations
        
        for ite in range(self.iters):

            p = self.clf.predict_proba(X_test)[:,1]

            X_new = []    
            y_new = []
            
            if self.thrNormalize:
                thr = self.relabelThr * p.std()
            else:
                thr = self.relabelThr
            
            thr *= (self.decay ** ite)
            
            for idx in range(X_test.shape[0]):

                confidence = np.abs(p[idx] - 0.5)
                
                #w = int(np.round(2 * self.relabelWeight * confidence))
                #w = int(self.relabelWeight * sigmoid(10*(confidence - 0.2)))
                w = int(self.relabelWeight * (confidence > thr))
                
                X_new += [X_test[idx, ...]] * w
                y_new += [np.round(p[idx])] * w
                
            if self.substitute:
                X_aug = np.array(X_new)
                y_aug = np.array(y_new)
            else:
                X_aug = np.concatenate([X, X_new]) 
                y_aug = np.concatenate([y, y_new])                

            self.clf.fit(X_aug, y_aug)
            
    def predict_proba(self, X):
        
        p = self.clf.predict_proba(X)
        
        return p

    def predict(self, X):
        
        c = (self.predict_proba(X)[:, 1] > 0.5)
        return c.astype(int)