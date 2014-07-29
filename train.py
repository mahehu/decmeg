# -*- coding: utf-8 -*-

"""DecMeg2014 example code.

  Simple prediction of the class labels of the test set by:
- pooling all the triaining trials of all subjects in one dataset.
- Extracting the MEG data in the first 500ms from when the
  stimulus starts.
- Using a linear classifier (logistic regression).
"""

# TODO: try to add filtering
# - change number of blocks (now 40)
# - change starting time (now 0.2 s)

# TODO: try other classifiers: random forest, extratrees

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from scipy.io import loadmat, savemat
from scipy.signal import lfilter
import time
import sys
import copy 
import scipy
import datetime 

from LrCollection import LrCollection
from IterativeTrainer import IterativeTrainer

def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def polarCoordinates(X):
    
    Xp = np.zeros_like(X)

    Xm = X[:, 2::3, :]
    Xg = X[:, 0::3, :] + 1j * X[:, 1::3, :]

    Xp[:, 2::3, :] = Xm
    Xp[:, 0::3, :] = np.abs(Xg)
    Xp[:, 1::3, :] = np.unwrap(np.angle(Xg))
    
    return Xp

def loadData(filename, 
             dsMode = "decimate", 
             downSample = 8, 
             start = 130, 
             stop = 375, 
             clipThr = None,
             numSensors = 306):

    g1Channels = [i for i in range(306) if i%3 == 0] # Gradiometer 1
    g2Channels = [i for i in range(306) if i%3 == 1] # Gradiometer 2
    g3Channels = [i for i in range(306) if i%3 == 2] # Magnetometer
    
    sys.stdout.write("Loading " + filename + "...")
    sys.stdout.flush()
    data = loadmat(filename, squeeze_me=True)
    XX = data['X']

    # Sensors ordered by their location (from back to front)

    sensorLocations = loadmat("sensorsFromBack.mat")
    idx = sensorLocations["idx"] - 1
    idx = idx.ravel()
    idx = idx[:numSensors]
    
    # Remove type 1 gradiometers
    #idx = [i for i in range(numSensors) if idx[i] not in g1Channels]
    
    XX = XX[:, idx, :]    
    
    try:
        yy = data['y']
    except:
        yy = None

    try:
        ids = data['Id']
    except:
        ids = None

    if dsMode == "block":
        XX = lfilter(np.ones(downSample,), 1, XX)
        XX = XX[:,:,::downSample]
    else:
        XX = scipy.signal.decimate(XX, downSample)

    XX_calib = XX[...,:start]
    XX = XX[..., int(start / float(downSample) + 0.5):int(stop / float(downSample) + 0.5)]

    X_new = copy.deepcopy(XX)

    for trial in range(XX.shape[0]):
	for t in range(XX.shape[2]):
	    X_new[trial,:,t] = X_new[trial,:,t] / mad(X_new[trial,:,t])
            X_new[trial,:,t] = X_new[trial,:,t] - np.median(X_new[trial,:,t])

	X_new[trial,...] = X_new[trial,...] / mad(XX_calib[trial,:,:start])
	X_new[trial,...] = X_new[trial,...] - np.median(XX_calib[trial,:,:start])

    XX = np.concatenate((XX, X_new), axis = 1)
    #XX = X_new

    if clipThr is not None:
        numClipped = 0
        
        for n in range(XX.shape[0]):
            
            trial = XX[n,...]
            
            for channels in [g1Channels, g2Channels, g3Channels]:
                
                view = trial[channels, :]        
                numClipped += np.count_nonzero(np.abs(view) > clipThr*np.std(view))
    
                # Clip outliers
                view[view > clipThr*np.std(view)] = clipThr*np.std(view)
                view[view < -clipThr*np.std(view)] = -clipThr*np.std(view)
                
                trial[channels, :] = view
            
            XX[n,...] = trial

        sys.stdout.write("Clipped %.2f %%\n" % (100. * numClipped / np.size(XX)))
        sys.stdout.flush()        
    else:
        print
        
    XX = XX - np.mean(XX, axis = 0)
    XX = XX / np.std(XX, axis = 0)
    
    return XX, yy, ids
    
def run(C = 0.1, 
        relabelThr = 1.0, 
        relabelWeight = 1,
        width = 1, 
        downSample = 8, 
        start = 50, 
        stop = 375, 
        clipThr = 3.,
        numSensors = 306,
        substitute = False):

    print "DecMeg2014: https://www.kaggle.com/c/decoding-the-human-brain"
    print
    subjects_train = range(1, 17) # use range(1, 17) for all subjects
    print "Training on subjects", subjects_train 
        
    print "Downsampling with factor %d." % (downSample)

    X_train = []
    y_train = []
    X_test = []
    ids_test = []
    labels = []

    print
    print "Creating the training set."
    
    for subject in subjects_train:
        
        filename = 'data/train_subject%02d.mat' % subject

        XX, yy, ids = loadData(filename, start = start, stop = stop, numSensors = numSensors)

        X_train.append(XX)
        y_train.append(yy)
        labels = labels + ([subject] * yy.shape[0])

    X = np.vstack(X_train)    
    y = np.concatenate(y_train)

    print "Trainset:", X.shape

    subjects_test = range(17,24)
    ids_test = []
    labels_test = []
    
    for subject in subjects_test:

        filename = 'data/test_subject%02d.mat' % subject

        XX, yy, ids = loadData(filename, start = start, stop = stop, numSensors = numSensors)

        X_test.append(XX)
        labels_test = labels_test + ([subject] * XX.shape[0])
        ids_test.append(ids)

    X_test = np.vstack(X_test)
    ids_test = np.concatenate(ids_test)
    print "Testset:", X_test.shape
    
    scores = []

    n_jobs = 1
    print "Start training at %s." % (datetime.datetime.now())

#    baseClf = LrCollection(clf1=LogisticRegression(C = C, penalty = 'l2'), 
#                        clf2 = LogisticRegression(C = 1, penalty = 'l1'), 
#                        n_jobs = n_jobs,
#                        useCols = True, 
#                        useRows = True)
                        
    baseClf = LrCollection(clf1 = LogisticRegression(C = C, penalty = 'l2'), 
                        clf2 = RandomForestClassifier(n_estimators=1000, n_jobs = n_jobs),
                        n_jobs = n_jobs,
                        useCols = True, 
                        useRows = True)
                        
    clf = IterativeTrainer(clf = baseClf, 
                           relabelWeight = relabelWeight, 
                           relabelThr = relabelThr,
                           thrNormalize = False,
                           iters = 1,
                           substitute = substitute)
    
    for s, leaveOutSubject in enumerate(subjects_train):

        startTime = time.time()
        
        trainIdx = [i for i in range(len(labels)) if labels[i] != leaveOutSubject]
        testIdx  = [i for i in range(len(labels)) if labels[i] == leaveOutSubject]
        
        trainLabels = [l for l in labels if l != leaveOutSubject]
        testLabels  = [l for l in labels if l == leaveOutSubject]
        
        clf.fit(X[trainIdx, :, :], y[trainIdx], X[testIdx,:,:])        
        yHat = clf.predict(X[testIdx,:,:])
        
        score = np.mean(yHat == y[testIdx])
        scores = scores + [score]
        
        msg = "LOSO score for test subject %d is %.4f [%.1f min]. Mean = %.4f +- %.4f." % (leaveOutSubject, score, (time.time() - startTime)/60, np.mean(scores), np.std(scores))
        print msg

        filename_log = "results/log-last-daymmc4-2-both1.txt"
        with open(filename_log, "a") as f:
            f.write(msg + "\n")

    print "Mean score over all subjects is %.4f." % (np.mean(scores))

    ##### PREDICT HERE
    
    filename_submission = "submissions/submission-last-daymmc4-2-both1.txt"
    print "Creating submission file", filename_submission
    
    with open(filename_submission, "w") as f:
        f.write("Id,Prediction\n")
        
    for subject in subjects_test:
        
        idx = [i for i in range(len(ids_test)) if ids_test[i]/1000 == subject]
        
        X_subj = X_test[idx,...]
        id_subj = ids_test[idx]
            
        print "Fitting full model for test subject %d." % (subject)
        clf.fit(X, y, X_subj)
        
        print "Predicting."               
        y_subj = clf.predict(X_subj)

        with open(filename_submission, "a") as f:
            for i in range(y_subj.shape[0]):
                f.write("%d,%d\n" % (id_subj[i], y_subj[i]))

    print "Done."

if __name__ == "__main__":
    
    C = 10. ** -2.25
    relabelWeight = 10
    relabelThr = 0.1
    width = 1

    downsample = 8
    start = 130
    stop = 375

    numSensors = 306
    clipThr = None

    relabelThr = 0.1

    start = 130
    numSensors = 153
    substitute = True
        
    print "Training subst = %d, numSensors = %d, relabelWeight = %.1f downsample = %d, start = %d and relabelThr = %.1f." % (int(substitute), numSensors, relabelWeight, downsample, start, relabelThr)
    run(C, relabelThr, relabelWeight, width, downsample, start = start, stop = stop, clipThr = clipThr, numSensors = numSensors, substitute = substitute)

