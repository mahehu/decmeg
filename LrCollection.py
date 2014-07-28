# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 13:11:43 2014

@author: hehu
"""

from sklearn.linear_model import LogisticRegression
import numpy as np
import copy
import datetime
from sklearn.lda import LDA

class LrCollection():
    def __init__(self, clf1 = LogisticRegression(penalty = 'l1', C = 0.1), 
                 clf2 = LogisticRegression(penalty = 'l1', C = 0.1), 
                 n_jobs = 1,
                 useCols = True, 
                 useRows = True):
        
        if not isinstance(clf1, list):
            self.clf1 = [clf1]
        else:
            self.clf1 = clf1

        if not isinstance(clf2, list):
            self.clf2 = [clf2]
        else:
            self.clf2 = clf2

        self.n_jobs = n_jobs
        self.useCols = useCols
        self.useRows = useRows
        
    def getView(self, X, idx):

        cols = idx[1]
        rows = idx[2]
        
        result = X[idx[0], cols[0]:cols[1], rows[0]:rows[1]]
        result = np.reshape(result, (result.shape[0], -1))
        return result        
        
    def fit(self, X, y):
        
        self.classifiers = []
        
        # Generate column and row views 
                
        trials = range(X.shape[0])
                
        if self.useCols:
            for col in range(X.shape[1]):
                
                for c in self.clf1:
                    clf = copy.deepcopy(c)
                    rows = [0, X.shape[2]]
                    cols = [col, col+1]
               
                    view = self.getView(X, [trials, cols, rows])
                
                    clf.fit(view, y)
                    self.classifiers.append((clf, (cols, rows)))
                
        if self.useRows:
            
            for row in range(X.shape[2]):

                for c in self.clf1:
                    clf = copy.deepcopy(c)
                
                    rows = [row, row+1]
                    cols = [0, X.shape[1]]
                    view = self.getView(X, [trials, cols, rows])
    
                    clf.fit(view, y)
                    self.classifiers.append((clf, (cols, rows)))
               
        # Train second layer to merge the inputs.
            
        yHat = self.predict_proba_l1(X)

        for c in self.clf2:
            c.fit(yHat, y)
        
    def predict_proba_l1(self, X):
        
        yHat = []
        
        trials = range(X.shape[0])
        
        for classifier in self.classifiers:
            c = classifier[0]
            idx = classifier[1]
            view = self.getView(X, [trials, idx[0], idx[1]])

            p = c.predict_proba(view)[:, 1]
            yHat.append(p)
            
        return np.array(yHat).T
    
    def predict_proba(self, X):
        
        yHat = self.predict_proba_l1(X)            
        
        y = []
        for c in self.clf2:
            if y == []:
                y = c.predict_proba(yHat)
            else:
                y = y + c.predict_proba(yHat)

        y = y / len(self.clf2)
        
        return y
            
    def predict(self, X):

        yHat = (self.predict_proba(X) > 0.5)
        return yHat
        
        
