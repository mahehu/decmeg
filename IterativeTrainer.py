"""
IterativeTrainer.py

   Heikki.Huttunen@tut.fi, Jul 29th, 2014

   Defines the class IterativeTrainer, which iterates training and
   augmentation of the training set with test samples and their
   predicted labels.
   
===
Copyright (c) 2014, Heikki Huttunen 
Department of Signal Processing
Tampere University of Technology
Heikki.Huttunen@tut.fi

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Tampere University of Technology nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
    
class IterativeTrainer():
    
    def __init__(self, 
                 clf,
                 iters = 1, 
                 relabelWeight = 10,
                 relabelThr = 0.1,
                 substitute = False):
        """
        Initialize the IterativeTrainer. 
        
        Parameters:
        
            clf:            The classifier to iterate
            iters:          Number of training/augmentation iterations
            relabelWeight:  Duplication factor of included test samples 
                            (only used if substitute = False)        
            relabelThr:     Threshold for accepting predicted test samples in 
                            second iteration (only used if substitute = False)
            substitute:     If True, original training samples are discarded
                            on second training iteration. Otherwise test
                            samples are appended to training data.
                            
        Returns:
        
            self
            
        """
    
        self.clf = clf
        self.iters = iters
        self.relabelWeight = relabelWeight
        self.relabelThr = relabelThr
        self.substitute = substitute
        
    def fit(self, X, y, X_test):
        """
        Train a classifier using the iterative semisupervised approach.
        
        Parameters:
        
            X:      Training data (a numpy array of shape (n, p, t))
            y:      Training data labels (shape (n,))
            X_test: Test samples (shape (n_t, p, t))
            
        Returns:
        
            self
            
        """
        
        # 0'th iteration: fit with training data only
        
        self.clf.fit(X, y)
        
        # Run relabeling iterations
        
        for ite in range(self.iters):

            # Predict class probabilities for unlabeled test samples:

            p = self.clf.predict_proba(X_test)[:,1]

            X_new = []    
            y_new = []
            
            for idx in range(X_test.shape[0]):

                # "Confidence" is the distance from 0.5
                # w contains the number of times each test sample is
                # included in the new training data.
                # w is zero for samples below confidence threshold.

                confidence = np.abs(p[idx] - 0.5)
                w = int(self.relabelWeight * (confidence > self.relabelThr))
                
                X_new += [X_test[idx, ...]] * w
                y_new += [np.round(p[idx])] * w
              
            if self.substitute:

                # The training set is completely substituted with the test samples.
            
                X_aug = np.array(X_new)
                y_aug = np.array(y_new)

            else:
                
                # The training set augmented with the found test samples.

                X_aug = np.concatenate([X, X_new]) 
                y_aug = np.concatenate([y, y_new])                

            # Train a model with the augmented training set.

            self.clf.fit(X_aug, y_aug)
            
    def predict_proba(self, X):
        """
        Predict class membership probabilities.
        
        Parameters:
        
            X:      Test samples (shape (n_t, p, t))
            
        Returns:
        
            Numpy array of probabilities (shape: (n_t, num_classes))
            
        """
        
        return self.clf.predict_proba(X)
        
    def predict(self, X):
        """
        Predict class labels for the test data.
        
        Parameters:
        
            X:      Test samples (shape (n_t, p, t))
            
        Returns:
        
            Numpy array of class labels (shape: (n_t,))
            
        """
        c = (self.predict_proba(X)[:, 1] > 0.5)
        return c.astype(int)
        