"""
   DecMeg2014 2nd place submission code. 

   Heikki.Huttunen@tut.fi, Jul 29th, 2014

   The model is a hierarchical combination of logistic regression and 
   random forest. The first layer consists of a collection of 337 logistic 
   regression classifiers, each using data either from a single sensor 
   (31 features) or data from a single time point (306 features). The 
   resulting probability estimates are fed to a 1000-tree random forest, 
   which makes the final decision. 
   
   The model is wrapped into the LrCollection class.
   The prediction is boosted in a semisupervised manner by
   iterated training with the test samples and their predicted classes
   only. This iteration is wrapped in the class IterativeTrainer.
   
   Requires sklearn, scipy and numpy packages.
   
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

# Generic imports 

import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter, decimate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import time
import sys
import copy 
import datetime 
import os.path
import json
import cPickle as pickle

from LrCollection import LrCollection
from IterativeTrainer import IterativeTrainer

def loadData(filename,
             downsample = 8, 
             start = 130, 
             stop = 375,
	     numSensors = 306):
    """
    Load, downsample and normalize the data from one test subject.
    
    Args:

        filename:   input mat file name
        downsample: downsampling factor
        start:      first time index in the result array (in samples)
        stop:       last time index in the result array (in samples)
	numSensors: number of sensors to use
    
    Returns: 
        
        X:          the 3-dimensional input data array
        y:          class labels (None if not available)
        ids:        the sample Id's of the samples, e.g., 17000, 17001, ...
                    (None if not available)
        
    """
    
    print "Loading " + filename + "..."
    data = loadmat(filename, squeeze_me=True)
    X = data['X']
   
    # Sort sensors ordered by their location (from back to front)
    # The mat file is generated from file NeuroMagSensorsDeviceSpace.mat
    # provided by the organizers.

    sensorLocations = loadmat("sensorsFromBack.mat")
    idx = sensorLocations["idx"] - 1
    idx = idx.ravel()
    idx = idx[:numSensors]
    X = X[:, idx, :]

    # Class labels available only for training data.
   
    try:
        y = data['y']
    except:
        y = None

    # Ids available only for test data

    try:
        ids = data['Id']
    except:
        ids = None

    # Decimate the time dimension (lowpass filtering + resampling)

    X = decimate(X, downsample)
    
    # Extract only the requested time span
    
    startIdx = int(start / float(downsample) + 0.5)
    stopIdx  = int(stop / float(downsample) + 0.5)
    X = X[..., startIdx:stopIdx]

    # Normalize each measurement

    X = X - np.mean(X, axis = 0)
    X = X / np.std(X, axis = 0)
    
    return X, y, ids
    
def run(datapath = "data",
	modelpath = "models",
	testdatapath = "data",
        C = 0.1, 
        numTrees = 1000,
        downsample = 8, 
        start = 130, 
        stop = 375, 
        numSensors = 306,
        relabelThr = 1.0, 
        relabelWeight = 1,
        iterations = 1,
        substitute = True,
        estimateCvScore = True):
    """
    Run training and serialize trained models.
    
    Args:
    
        datapath:        subfolder where the training .mat files are located. 
        modelpath: subfolder where trained models are stored
        testdatapath: subfolder where testing .mat files are located
        C:               Regularization parameter for logistic regression
        numTrees:        Number of trees in random forest
        downsample:      Downsampling factor in preprocessing
        start:           First time index in the result array (in samples)
        stop:            Last time index in the result array (in samples)
        numSensors:      Number of sensors to use (starting from back of the head)
        relabelThr:      Threshold for accepting predicted test samples in 
                         second iteration (only used if substitute = False)
        relabelWeight:   Duplication factor of included test samples 
                         (only used if substitute = False)
        substitute:      If True, original training samples are discarded
                         on second training iteration. Otherwise test
                         samples are appended to training data.    
        estimateCvScore: If True, we do a full 16-fold CV for each training 
                         subject. Otherwise only final submission is created.
     
    Returns:
    
        Nothing.
        
    """

    print "DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain"
    print "[2nd place submission. Heikki.Huttunen@tut.fi]"
    
    subjects_train = range(1, 17) # use range(1, 17) for all subjects
    print "Training on subjects", subjects_train 
    
    X_train = []        # The training data
    y_train = []        # Training labels
    X_test = []         # Test data
    ids_test = []       # Test ids
    labels = []         # Subject number for each trial in training data
    labels_test = []    # Subject number for each trial in test data

    print "Loading %d train subjects." % (len(subjects_train))
    
    for subject in subjects_train:
        
        filename = os.path.join(datapath, 'train_subject%02d.mat' % subject)

        XX, yy, ids = loadData(filename = filename, 
                               downsample = downsample,
                               start = start, 
                               stop = stop)

        X_train.append(XX)
        y_train.append(yy)
        labels = labels + ([subject] * yy.shape[0])

    X = np.vstack(X_train)    
    y = np.concatenate(y_train)

    print "Training set size:", X.shape

    subjects_test = range(17,24)
    
    print "Loading %d test subjects." % (len(subjects_test))
        
    for subject in subjects_test:

        filename = os.path.join(testdatapath, 'test_subject%02d.mat' % subject)

        XX, yy, ids = loadData(filename = filename, 
                               downsample = downsample,
                               start = start, 
                               stop = stop)

        X_test.append(XX)
        labels_test = labels_test + ([subject] * XX.shape[0])
        ids_test.append(ids)

    X_test = np.vstack(X_test)
    ids_test = np.concatenate(ids_test)
    print "Testset:", X_test.shape

    # Define our two-layer classifier

    clf1 = LogisticRegression(C = C, penalty = 'l2')
    clf2 = RandomForestClassifier(n_estimators = numTrees, n_jobs = 1)
    
    baseClf = LrCollection(clf1 = clf1, 
                           clf2 = clf2,     # useCols and useRows can be used
                           useCols = True,  # for predicting with only time 
                           useRows = True)  # or sensor dimension

    # Wrap the classifier inside our iterative scheme.
    # The clf argument accepts any sklearn classifier, as well.
                        
    clf = IterativeTrainer(clf = baseClf, 
                           relabelWeight = relabelWeight, 
                           relabelThr = relabelThr,
                           iters = 1,
                           substitute = substitute)

    if estimateCvScore:

        # Store leave-one-subject-out (LOSO) scores here
    
        scores = []    
    
        # Train 16 times with one subject excluded each time.
    
        print "Start LOSO training at %s." % (datetime.datetime.now())
        
        for leaveOutSubject in subjects_train:
    
            startTime = time.time()
            
            # Choose training and test indices
            
            trainIdx = [i for i in range(len(labels)) if labels[i] != leaveOutSubject]
            testIdx  = [i for i in range(len(labels)) if labels[i] == leaveOutSubject]
            
            # Train the model. Note that the test data are also passed
            # to training due to iterative transduction.
            
            clf.fit(X[trainIdx, :, :], y[trainIdx], X[testIdx,:,:])        

            # Predict classes for left-out subject

            yHat = clf.predict(X[testIdx,:,:])

            # Estimate accuracy.

            score = np.mean(yHat == y[testIdx])
            scores = scores + [score]
            
            print "LOSO score for test subject %d is %.4f [%.1f min]. \
                   Mean = %.4f +- %.4f." % \
                   (leaveOutSubject, 
                    score, 
                    (time.time() - startTime)/60, 
                    np.mean(scores), 
                    np.std(scores))
    
        print "Mean score over all subjects is %.4f." % (np.mean(scores))

    # Train the model.

    print "Start training final models at %s." % (datetime.datetime.now())
    
    # Make sure there is a directory for storing the models:

    if not os.path.exists(modelpath):
        os.makedirs(modelpath)

    # Train a subjective model for each test subject, and serialize them.
    
    for subject in subjects_test:
        
        filename = "model%d.pkl" % subject
        output = open(os.path.join(modelpath, filename), "wb")

        # Find trials for this test subject:
        
        idx = [i for i in range(len(ids_test)) if ids_test[i] / 1000 == subject]
        
        X_subj = X_test[idx,...]
        id_subj = ids_test[idx]
        
        # Fit a model for predicting labels for one subject.
        # Note that the test data are also passed to training due 
        # to iterative transduction.
            
        print "Fitting full model for test subject %d." % (subject)
        clf.fit(X, y, X_subj)
        
        print "Writing %s..." % os.path.join(modelpath, filename)
        pickle.dump(clf, output)

        output.close()
        
    print "Done."

if __name__ == "__main__":
    """
    Set training parameters and call main function.
    """
    
    f = open("SETTINGS.json", "r")
    settings = json.load(f)
    datapath = settings["TRAIN_DATA_PATH"]
    modelpath = settings["MODEL_PATH"]
    testdatapath = settings["TEST_DATA_PATH"]
    f.close()
    C = 10. ** -2.25
    numTrees = 1000
    relabelWeight = 10
    relabelThr = 0.1
    downsample = 8
    start = 130
    stop = 375
    substitute = True
    estimateCvScore = False
    iterations = 1
    numSensors = 306

    try:
        randomseed = int(sys.argv[1])
    catch:
    	randomseed = 3
    	
    np.random.seed(randomseed)
        
    run(datapath = datapath,
        modelpath = modelpath,
        testdatapath = testdatapath,
        C = C, 
        numTrees = numTrees,
        relabelThr = relabelThr, 
        relabelWeight = relabelWeight, 
        iterations = iterations,
        downsample = downsample, 
        start = start, 
        stop = stop, 
        numSensors = numSensors,
        substitute = substitute,
        estimateCvScore = estimateCvScore,
	randomseed = randomseed)


