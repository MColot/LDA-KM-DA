"""
Evaluation of LDA-KM-DA and state-of-the-art methods for cross-subject hand gesture recognition
with 3 public datasets using Time Domain Features and Riemmannian features

complete the code with the path to the folders where the datasets have been placed
Change the parameters to test different datasets, features sets, and models

additional dependency = pyriemann

Author: Martin Colot
"""
from pyriemann.estimation import XdawnCovariances, Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

from dataLoader import load_ninaproDB4, load_ToroOssaba, load_EMG_EPN_612
from timeDomainFeatures import computeTDF, normalizeTDF

# ------------ PARAMETERS ------------------------------------------------------------------------------

# Edit these paths for the chosen datasets
pathToNinaproDB4 = None
pathToToroOssaba = None
pathToEmgEpn612 = None

datasets = ["ninaproDB4", "ToroOssaba", "EMG_EPN_612"]
testDataset = datasets[0]  # Edit this to change the test dataset

featureSets = ["TDF", "CMTS", "Xdawn CMTS"]
testFeatures = featureSets[0]  # Edit this to change the feature sets used

models = {0: "Intra-subject",
          1: "Cross-subject",
          2: "Normalization",
          3: "MDD",
          4: "CDEM",
          5: "Simplified CDEM",
          6: "VADA",
          7: "DIRT-T",
          8: "LDA-KM-DA",
          9: "MDD -> LDA-KM-DA",
          10: "Simplified CDEM -> LDA-KM-DA"}
testModels = [models[0], models[1], models[2], models[8]]  # change this to choose which model to evaluate

# -------------- EVALUATION ----------------------------------------------------------------------------

X, Y = None, None
if testDataset == "ninaproDB4":
    X, Y = load_ninaproDB4(pathToNinaproDB4)
elif testDataset == "ToroOssaba":
    X, Y = load_ToroOssaba(pathToToroOssaba)
elif testDataset == "EMG_EPN_612":
    X, Y = load_EMG_EPN_612(pathToEmgEpn612)
assert X is not None and Y is not None


tdfX = computeTDF(X)  # TDF features
tdfX_n = normalizeTDF(tdfX)  # normalized TDF features

pipelineCMTS = Pipeline(
            [('cov', Covariances(estimator='oas')),
             ('ts', TangentSpace('riemann'))])
pipelineXdawnCMTS = Pipeline(
            [('cov', XdawnCovariances(nfilter=3, estimator='oas')),
             ('ts', TangentSpace('riemann'))])
cmtsX = np.array([pipelineCMTS.fit_transform(X[s]) for s in range(len(X))], dtype=object) # normalized CMTS features

scores = dict()

if "Intra-subject" in testModels:
    score = []
    for s in range(len(X)):
        features = None
        if testFeatures == "TDF":
            score.append(cross_val_score(LogisticRegression('l2', solver="liblinear"), tdfX_n[s], Y[s].astype(int), cv=KFold(n_splits=5, shuffle=False)))
        elif testFeatures == "CMTS":
            score.append(cross_val_score(LogisticRegression('l2', solver="liblinear"), cmtsX[s], Y[s].astype(int), cv=KFold(n_splits=5, shuffle=False)))
        elif testFeatures == "Xdawn CMTS":
            pipeline = Pipeline(
                [('cov', XdawnCovariances(nfilter=3, estimator='oas')),
                 ('ts', TangentSpace('riemann')),
                 ("lr", LogisticRegression('l2', solver="liblinear"))])
            score.append(cross_val_score(pipeline, X[s], Y[s].astype(int), cv=KFold(n_splits=5, shuffle=False)))
        print(s, score[-1])
    print("average score Intra-subject :", np.mean(score))
    scores["Intra-subject"] = score

if "Cross-subject" in testModels:
    pass

if "Normalization" in testModels:
    pass

if "MDD" in testModels:
    pass

if "CDEM" in testModels:
    pass

if "Simplified CDEM" in testModels:
    pass

if "VADA" in testModels:
    pass

if "DIRT_T" in testModels:
    pass

if "LDA-KM-DA" in testModels:
    pass

if "MDD -> LDA-KM-DA" in testModels:
    pass

if "Simplified CDEM -> LDA-KM-DA" in testModels:
    pass