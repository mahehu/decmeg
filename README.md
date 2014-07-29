decmeg
======
#Description

2nd place submission to the MEG decoding competition https://www.kaggle.com/c/decoding-the-human-brain

        [https://www.kaggle.com/c/decoding-the-human-brain]

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

   Example usage:

```
python train.py
```
   
#License

```
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
```