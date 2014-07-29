"""
LrCollection.py

   Heikki.Huttunen@tut.fi, Jul 29th, 2014

   Defines the class LrCollection: A hierarchical two-layer
   structure for MEG decoding. The input consists of time slices
   and sensor slices. Each 1st layer classifier will see one
   slice of the data either in time or sensor dimension.
   
   The 1st layer predictions are combined together using
   a second layer classifier.
   
   Classifiers can be any classifier with sklearn interface.
   Alternatively, there can be a list of classifiers for
   both layers, in which case all will be used.
   
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
import copy

class LrCollection():
    def __init__(self, 
                 clf1, 
                 clf2, 
                 useCols = True, 
                 useRows = True):
        """
        Initialize the LR collection classifier.
        
        Args:
        
            clf1:           First layer classifier. Can also be a list of 
                            classifiers.
            clf2:           Second layer classifier. Can also be a list of 
                            classifiers. In this case, the output will be
                            averaged over the predictions of the list.
            useCols:        If true, train a predictor for each sensor
            useRows:        If true, train a predictor for each timepoint
            
        Returns:
        
            self
            
        """

        # If the first layer classifier is not inside a list, store it
        # in a 1-element list.

        if not isinstance(clf1, list):
            self.clf1 = [clf1]
        else:
            self.clf1 = clf1

        # If the second layer classifier is not inside a list, store it
        # in a 1-element list.

        if not isinstance(clf2, list):
            self.clf2 = [clf2]
        else:
            self.clf2 = clf2

        self.useCols = useCols
        self.useRows = useRows
        
    def getView(self, X, idx):
        """
        Extract data from a single row or column in the data matrix.
        Can also span multiple rows/columns.
        
        Args:
        
            X:      Input data array of shape (n, p, t)
            idx:    A three element list of requested slice coordinates.
                    The element idx[0] is the list of trials to extract.
                    Second element idx[1] is a two element vector with
                    start and end indices in the sensor space (e.g.,
                    idx[1] = [0, 306]). The third element idx[1] is a 
                    two element vector with start and end indices in the time
                    dimension (e.g., idx[1] = [0, 31]).
        """
        
        cols = idx[1]
        rows = idx[2]
        
        # Extract the requested slice and reshape to a design matrix
        # of shape (n, cols[1]-cols[0], rows[1]-rows[0]).
        
        result = X[idx[0], cols[0]:cols[1], rows[0]:rows[1]]
        result = np.reshape(result, (result.shape[0], -1))

        return result        
        
    def fit(self, X, y):
        """
        Train the hierarchical classification model.
        
        Args:
        
            X:      Input training data array of shape (n, p, t)
            y:      Training class labels (shape: (n,))
            
        """
        
        # All 1st layer classifiers will be stored here:
        
        self.classifiers = []

        # We will be using all data for training.

        trials = range(X.shape[0])
        
        # Generate column and row views for the selected trials.
                
        if self.useCols:
            
            for col in range(X.shape[1]):
                
                # Train all classifiers in the list of 1st layer
                
                for c in self.clf1:
                    clf = copy.deepcopy(c)
                    
                    # Define the slice with one sensor and all time indices
                    
                    rows = [0, X.shape[2]]
                    cols = [col, col+1]
               
                    # Get the data for this view
               
                    view = self.getView(X, [trials, cols, rows])

                    # Train a classifier with this view.
                
                    clf.fit(view, y)
                    self.classifiers.append((clf, (cols, rows)))
                
        if self.useRows:
            
            for row in range(X.shape[2]):

                # Train all classifiers in the list of 1st layer
                
                for c in self.clf1:
                    clf = copy.deepcopy(c)
                
                    # Define the slice with one time index and all sensors
                
                    rows = [row, row+1]
                    cols = [0, X.shape[1]]
                    
                    # Get the data for this view
                    view = self.getView(X, [trials, cols, rows])
    
                    # Train a classifier with this view.
    
                    clf.fit(view, y)
                    self.classifiers.append((clf, (cols, rows)))
               
        # The input to the second layer are the predicted probabilities
        # from the first layer predictors.
            
        yHat = self.predict_proba_l1(X)

        # Train second layer classifiers to merge the inputs.
        
        for c in self.clf2:
            c.fit(yHat, y)
        
    def predict_proba_l1(self, X):
        """
        Predict class probabilities for the test data using all 1st layer 
        classifiers.
        
        Args:
        
            X:      Input test data array of shape (n_t, p, t)
            
         Returns:
        
            Numpy array of probabilities (shape: (n_t, num_classifiers))
            
        """
        
        yHat = []
        
        trials = range(X.shape[0])
        
        # Predict probability with every classifier in our list
        
        for classifier in self.classifiers:
            
            # classifier is a list with the "classifier" 
            # and the related "indices".
            
            c = classifier[0]
            idx = classifier[1]
            
            # Get the same view as in training stage
            
            view = self.getView(X, [trials, idx[0], idx[1]])

            # Predict

            p = c.predict_proba(view)[:, 1]
            yHat.append(p)

        return np.array(yHat).T
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the test data.
        
        Args:
        
            X:      Input test data array of shape (n_t, p, t)
            
         Returns:
        
            Numpy array of probabilities (shape: (n_t, num_classes))
            
        """
        
        # Predict first using all 1st layer classifiers:
        
        yHat = self.predict_proba_l1(X)            

        # Combine these together with each of the 2nd layer classifiers
        
        y = []

        for c in self.clf2:

            if y == []:
                y = c.predict_proba(yHat)
            else:
                y = y + c.predict_proba(yHat)

        y = y / len(self.clf2)
        
        return y
            
    def predict(self, X):
        """
        Predict class labels for the test data.
        
        Args:
        
            X:      Input test data array of shape (n_t, p, t)
            
         Returns:
        
            Numpy array of probabilities (shape: (n_t, ))
            
        """
        yHat = (self.predict_proba(X) > 0.5)
        return yHat
                
