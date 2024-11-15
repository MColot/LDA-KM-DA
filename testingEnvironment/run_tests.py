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
from UDA.MDD import fit_MDD

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
            pipelineXdawnCMTS = Pipeline(
                [('cov', XdawnCovariances(nfilter=3, estimator='oas')),
                 ('ts', TangentSpace('riemann')),
                 ("lr", LogisticRegression('l2', solver="liblinear"))])
            score.append(cross_val_score(pipelineXdawnCMTS, X[s], Y[s].astype(int), cv=KFold(n_splits=5, shuffle=False)))
        print(s, score[-1])
    print("average score Intra-subject :", np.mean(score))
    scores["Intra-subject"] = score

if "Cross-subject" in testModels:
    score = []
    for test in range(len(X)):
        train = np.arange(len(X))
        train = np.delete(train, test)
        if testFeatures == "TDF":
            cl = LogisticRegression('l2', solver="liblinear")
            cl.fit(np.concatenate([x for x in tdfX[train]]),  np.concatenate([y for y in Y[train]]).astype(int))
            score.append(np.mean(Y[test].astype(int) == cl.predict(tdfX[test])))
        elif testFeatures == "CMTS":
            cl = Pipeline(
                [('cov', Covariances(estimator='oas')),
                 ('ts', TangentSpace('riemann')),
                 ('lr', LogisticRegression('l2', solver="liblinear"))])
            cl.fit(np.concatenate([x for x in X[train]]), np.concatenate([y for y in Y[train]]).astype(int))
            score.append(np.mean(Y[test].astype(int) == cl.predict(X[test])))
        elif testFeatures == "Xdawn CMTS":
            cl = Pipeline(
                [('cov', XdawnCovariances(nfilter=3, estimator='oas')),
                 ('ts', TangentSpace('riemann')),
                 ("lr", LogisticRegression('l2', solver="liblinear"))])
            cl.fit(np.concatenate([x for x in X[train]]), np.concatenate([y for y in Y[train]]).astype(int))
            score.append(np.mean(Y[test].astype(int) == cl.predict(X[test])))
        print(test, score[-1])
    print("average score Cross-subject :", np.mean(score))
    scores["Cross-subject"] = score

if "Normalization" in testModels:
    score = []
    for test in range(len(X)):
        train = np.arange(len(X))
        train = np.delete(train, test)
        if testFeatures == "TDF":
            cl = LogisticRegression('l2', solver="liblinear")
            cl.fit(np.concatenate([x for x in tdfX_n[train]]), np.concatenate([y for y in Y[train]]).astype(int))
            score.append(np.mean(Y[test].astype(int) == cl.predict(tdfX_n[test])))
        if testFeatures == "CMTS":
            cl = LogisticRegression('l2', solver="liblinear")
            cl.fit(np.concatenate([x for x in cmtsX[train]]), np.concatenate([y for y in Y[train]]).astype(int))
            score.append(np.mean(Y[test].astype(int) == cl.predict(cmtsX[test])))
        elif testFeatures == "Xdawn CMTS":
            cov = XdawnCovariances(nfilter=3, estimator='oas').fit(np.concatenate(X[train]), np.concatenate(Y[train].astype(int)))
            xTrain = [cov.transform(x) for x in X[train]]
            xTest = cov.transform(X[test])
            tsTrain = np.concatenate([TangentSpace('riemann').fit_transform(x) for x in xTrain])
            tsTest = TangentSpace('riemann').fit_transform(xTest)
            cl = LogisticRegression('l2', solver="liblinear").fit(tsTrain, np.concatenate(Y[train].astype(int)))
            score.append(np.mean(Y[test].astype(int) == cl.predict(tsTest)))
        print(test, score[-1])
    print("average score normalization :", np.mean(score))
    scores["Normalization"] = score

if "MDD" in testModels:
    score = []
    for test in range(len(X)):
        train = np.arange(len(X))
        train = np.delete(train, test)
        if testFeatures == "TDF":
            cl = fit_MDD(np.concatenate(tdfX_n[train]), np.concatenate(Y[train]).astype(int), tdfX_n[test], hidden=50, epochs=10)
            score.append(np.mean(Y[test].astype(int) == cl.predict(tdfX_n[test])))
        if testFeatures == "CMTS":
            cl = fit_MDD(np.concatenate(cmtsX[train]), np.concatenate(Y[train]).astype(int), cmtsX[test], hidden=50, epochs=10)
            score.append(np.mean(Y[test].astype(int) == cl.predict(cmtsX[test])))
        elif testFeatures == "Xdawn CMTS":
            cov = XdawnCovariances(nfilter=3, estimator='oas').fit(np.concatenate(X[train]), np.concatenate(Y[train].astype(int)))
            xTrain = [cov.transform(x) for x in X[train]]
            xTest = cov.transform(X[test])
            tsTrain = np.concatenate([TangentSpace('riemann').fit_transform(x) for x in xTrain])
            tsTest = TangentSpace('riemann').fit_transform(xTest)
            cl = fit_MDD(tsTrain, np.concatenate(Y[train]).astype(int), tsTest, hidden=50, epochs=10)
            score.append(np.mean(Y[test].astype(int) == cl.predict(tsTest)))
        print(test, score[-1])
    print("average score MDD :", np.mean(score))
    scores["MDD"] = score

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
