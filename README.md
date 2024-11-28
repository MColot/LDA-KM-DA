# LDA-KM-DA for Unsupervised Domain Adaptation

This directory contains the code for the paper "Linear Non-conservative Unsupervised Domain Adaptation Applied In Cross-Subject EMG Gesture Recognition"
Please refer to our paper for the technical details.

![graphicalAbtract_3_onlyPartB](https://github.com/user-attachments/assets/0d11dc41-161d-40c4-a4fa-beb5b69c1adc)

# Code and data

The proposed method, LDA-KM-DA, is implemented in ldaKmDa.py as a simple Python function that returns an sklearn pipeline, fitted for the target domain.

Evaluation of the proposed method and state-of-the-art methods discussed in the paper can be launched from run_tests.py. 
Download the public datasets and set the paths to their files in the Python script run_tests.py. 
Edit the testing parameters (models and features) in the Python script run_tests.py.

Methods to launch the 3 public datasets and extract features are implemented in testingEnvironment/dataLoader.py and testingEnvironment/timeDomainFeatures.py.
The state-of-the-art methods discussed in the paper (MDD, CDEM, Simplified CDEM, VADA, DIRT-T) are implemented in the testingEnvironment/UDA

# Results

The results obtained on the 3 public datasets that are presented in the paper can be reproduced using the code in this directory

![boxplotClassification4Datasets](https://github.com/user-attachments/assets/e118c64f-7f55-40f8-a1dc-b86a1f12c1b2)

# Dependencies
For MDD, install tf_keras
