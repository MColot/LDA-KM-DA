import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer to remap cluster labels
# Define a step to run KMeans predict and then remap
class KMeansWithMapping(BaseEstimator, TransformerMixin):
    def __init__(self, kmeans, class_mapping):
        self.kmeans = kmeans
        self.class_mapping = class_mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Predict clusters
        cluster_indices = self.kmeans.predict(X)
        # Map clusters to original class labels
        return np.vectorize(self.class_mapping.get)(cluster_indices)

    def predict(self, X):
        # Call transform to perform predict + mapping
        return self.transform(X)


def LDA_KM_DA(predInit, xTest, eps=0.01, yTest=None, maxEpochs=100, plotResults=False):
    """
    Implements the LDA-KM-DA algorithm for unsupervised structure-based non-conservative domain adaptation
    :param predInit: initial prediction of pseudo-labels (should be obtained by a source classifier)
    :param xTest: unlabeled target samples
    :param eps: epsilon value to determine convergence of the prediction
    :param yTest: labels of the target samples, used to print the evolution of the accuracy only
    :param maxEpochs: maximum number of iterations of the pseudo labels refinement
    :return: a fitted classifier, adapted for the target domain
    """
    projEvol = []
    predEvol = []
    pred = predInit
    lastPred = predInit
    diff = 1
    epochsCount = 0
    classes = np.unique(predInit)
    if yTest is not None:
        print("Initial accuracy :", np.mean(pred == yTest))
    while diff > eps and epochsCount < maxEpochs:
        try:
            nclasses = len(np.unique(pred))
            lda = LDA(n_components=nclasses - 1, shrinkage="auto", solver="eigen").fit(xTest, pred)
            newProj = lda.transform(xTest)
            pred = KMeans(n_clusters=nclasses, n_init=1,
                          init=[np.mean(newProj[pred == x], axis=0) for x in classes]).fit(newProj).labels_
            pred = np.array([c for c in classes])[pred]
            if plotResults:
                projEvol.append(newProj)
                predEvol.append(pred)
            diff = np.mean(lastPred != pred)
            lastPred = pred
            epochsCount += 1
            if yTest is not None:
                print(f"Adapted accuracy after {epochsCount} iterations :", np.mean(pred == yTest))
        except Exception as e:
            print(f"LDA-KM-DA has found less than {len(classes)} classes in the data. The adapted classifier might be wrong")
            break
    nclasses = len(np.unique(pred))
    finalLDA = LDA(n_components=nclasses - 1, shrinkage="auto", solver="eigen").fit(xTest, pred)
    finalProj = finalLDA.transform(xTest)
    finalKMeans = KMeans(n_clusters=nclasses, n_init=1,
                         init=[np.mean(finalProj[pred == x], axis=0) for x in classes]).fit(finalProj)
    clusterToClass = {i: classes[i] for i in range(len(classes))}
    cl = Pipeline([("lda", finalLDA),
                   ("K-means", KMeansWithMapping(kmeans=finalKMeans, class_mapping=clusterToClass))])

    if plotResults and yTest is not None:
        fig, ax = plt.subplots(2, len(predEvol), figsize=(5*len(predEvol), 10))
        for i in range(len(predEvol)):
            ax[0, i].scatter(projEvol[i][:, 0], projEvol[i][:, 1], c=predEvol[i])
            ax[1, i].scatter(projEvol[i][:, 0], projEvol[i][:, 1], c=yTest)
            ax[0, i].set_title(f"Iteration {i+1}\naccuracy={np.mean(predEvol[i] == yTest)*100}%")
            ax[1, i].set_title("Real labels")
        plt.show()
    return cl


if __name__ == "__main__":
    # test of LDA-KM-DA with synthetic data
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, cross_val_predict
    import pandas as pd
    import seaborn as sn
    from sklearn.preprocessing import StandardScaler, normalize
    from autorank import autorank, plot_stats

    nSubjects = 30

    x, y = make_classification(random_state=42,
                               n_samples=nSubjects * 1000,
                               n_features=20,
                               n_informative=10,
                               n_redundant=5,
                               n_repeated=5,
                               n_classes=3,
                               n_clusters_per_class=1,
                               class_sep=0.6) # change this value to modify the difficulty of the adaptation
    x = x.reshape((nSubjects, 1000, 20))
    y = y.reshape((nSubjects, 1000))

    # domain shift
    for i in range(nSubjects):
        for c in range(3):
            x[i][y[i] == c] += (np.random.random(20) * 2 - 1) * 2

    # normalization
    for i in range(len(x)):
        x[i] = normalize(x[i])
        x[i] = StandardScaler().fit_transform(x[i])

    lda = LDA(n_components=2, shrinkage="auto", solver="eigen").fit(np.concatenate(x), np.concatenate(y))

    p = lda.transform(np.concatenate(x))
    plt.scatter(p[:, 0], p[:, 1], c=np.concatenate(y))
    plt.show()

    fig, ax = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(5):
        p = lda.transform(x[i])
        ax[0, i].scatter(p[:, 0], p[:, 1], c=y[i])
        p = LDA(n_components=2, shrinkage="auto", solver="eigen").fit_transform(x[i], y[i])
        ax[1, i].scatter(p[:, 0], p[:, 1], c=y[i])
    plt.show()

    accIntra = []
    accCross = []
    accUDA = []
    for test in range(nSubjects):
        train = np.arange(nSubjects)
        train = np.delete(train, test)

        cl = LogisticRegression("l2", solver="liblinear").fit(np.concatenate(x[train]), np.concatenate(y[train]))
        pred = cl.predict(x[test])

        ldakmda = LDA_KM_DA(pred, x[test], eps=0.01, yTest=y[test], maxEpochs=100, plotResults=True)
        pred2 = ldakmda.predict(x[test])

        accCross.append(np.mean(y[test] == pred))
        accUDA.append(np.mean(y[test] == pred2))

        predIntra = cross_val_predict(LogisticRegression("l2", solver="liblinear"), x[test], y[test],
                                      cv=KFold(n_splits=5, shuffle=False))
        accIntra.append(np.mean(y[test] == predIntra))

        print(accCross[-1], " -> ", accUDA[-1], " / ", accIntra[-1])

    scoreDF = pd.DataFrame(
        {"Cross-subjects\nwith normalization": accCross, "LDA-KM-DA": accUDA, "Intra-subject": accIntra})
    sn.boxplot(scoreDF)
    plt.show()

    result = autorank(scoreDF, alpha=0.05, verbose=False)
    plot_stats(result)
    plt.show()
