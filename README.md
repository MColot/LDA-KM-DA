# LDA-KM-DA : Unsupervised Domain Adaptation for Cross-subject Gesture Recognition from EMG Signal

This directory contains the code for the paper "Linear Non-conservative Unsupervised Domain Adaptation Applied In Cross-Subject EMG Gesture Recognition"
Please refer to our paper for the technical details.

![graphicalAbtract_3_onlyPartB](https://github.com/user-attachments/assets/c958576f-b2df-455a-b45e-f17d21424b3e)

# Code and data

The proposed method, LDA-KM-DA, is implemented in ldaKmDa.py as a simple Python function that returns an sklearn pipeline, fitted for the target domain.

Evaluation of the proposed method and state-of-the-art methods discussed in the paper can be launched from run_tests.py. 
Download the public datasets and set the paths to their files in the Python script run_tests.py. 
Edit the testing parameters (models and features) in the Python script run_tests.py.

Methods to launch the 3 public datasets and extract features are implemented in testingEnvironment/dataLoader.py and testingEnvironment/timeDomainFeatures.py.
The state-of-the-art methods discussed in the paper (MDD, CDEM, Simplified CDEM, VADA, DIRT-T) are implemented in the testingEnvironment/UDA

# Results

The results obtained on the 3 public datasets that are presented in the paper can be reproduced using the code in this directory

![boxplotClassification4Datasets_whiteBackground](https://github.com/user-attachments/assets/c5e89513-351f-4923-ad87-2f7c7bab5e3c)

# Dependencies
For MDD, install tf_keras
