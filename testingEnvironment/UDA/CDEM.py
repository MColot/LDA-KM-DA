"""
Implementation of the CDEM algorithm in Python, based on the Matlab implementation available on : https://github.com/yuntaodu/CDEM
Adaptation of the code to implement the Simplified CDEM algorithm

Dependency : numpy, sklearn, scipy
Dependency for testing on synthetic data : pandas, seaborn, matplotlib, autorank

Author : Martin Colot
"""


import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.linalg import eigh


def constructW1(label):
    W = np.zeros((len(label), len(label)))
    num_class = np.max(label)
    for i in range(num_class):
        class_vector = (label == i).astype(np.float64)
        W += np.outer(class_vector, class_vector)
    return W


def L2Norm(x):
    l2_norm = np.sqrt(np.sum(x ** 2, axis=1)) + 1e-4
    y = x / l2_norm[:, np.newaxis]
    return y


def EuDist2(fea_a, fea_b, bSqrt=None):
    if bSqrt is None:
        bSqrt = 1

    D = None
    if fea_b is None:
        aa = np.sum(fea_a * fea_a, axis=1)
        ab = np.dot(fea_a, fea_a.T)
        if np.issparse(aa):
            aa = aa.toarray()
        D = np.add.outer(aa, aa) - 2 * ab
        D[D < 0] = 0

        if bSqrt:
            D = np.sqrt(D)

        D = np.maximum(D, D.T)
    else:
        aa = np.sum(fea_a * fea_a, axis=1)
        bb = np.sum(fea_b * fea_b, axis=1)
        ab = np.dot(fea_a, fea_b.T)
        D = np.add.outer(aa, bb) - 2 * ab
        D[D < 0] = 0
        if bSqrt:
            D = np.sqrt(D)

    return D


def constructP(Xs, Ys, Xt, Yt_pseudo, W, simplified, options):
    if "beta" not in options.keys():
        options["beta"] = 1
    if "lambda" not in options.keys():
        options["lambda"] = 1
    if "gamma" not in options.keys():
        options["gamma"] = 0.5
    if "eta" not in options.keys():
        options["eta"] = 0.0001
    if "sigma" not in options.keys():
        options["sigma"] = 0.1

    ns = Xs.shape[0]
    nt = Xt.shape[0]
    n = ns + nt
    C = len(np.unique(Ys))
    d = Xs.shape[1]

    e = np.concatenate((np.ones(ns) / ns, -np.ones(nt) / nt))
    M = options["lambda"] * np.outer(e, e) * C
    Qnew = np.zeros(ns + nt)

    Qcc_t = None  # Added

    if len(Yt_pseudo) > 0:
        idx = np.where(Yt_pseudo != -1)[0]
        Xt = Xt[idx, :]
        Yt_pseudo = Yt_pseudo[idx]
        nt = Xt.shape[0]
        n = ns + nt
        idx_s = np.arange(ns)
        W = W[np.ix_(np.concatenate((idx_s, ns + idx)), np.concatenate((idx_s, ns + idx)))]
        e = np.concatenate((np.ones(ns) / ns, -np.ones(nt) / nt))
        M = options["lambda"] * (np.outer(e, e)) * C

        unique_classes = np.unique(Ys)
        for c in unique_classes:
            e = np.zeros(n)
            ys_indices = np.where(Ys == c)[0]
            e[ys_indices] = 1 / len(ys_indices)
            yt_indices = np.where(Yt_pseudo == c)[0]
            e[ns + yt_indices] = -1 / len(yt_indices)
            e[np.isinf(e)] = 0
            nc = len(ys_indices) + len(yt_indices)
            M += options["lambda"] * nc * np.outer(e, e)

        tp = np.zeros((nt, C))
        for c in range(C):
            tp[Yt_pseudo == c, c] = 1
        tp2 = tp @ np.diag(1. / (1e-4 + sum(tp)))
        tp3 = tp2 @ tp.T
        Qcc_t = np.eye(nt) - tp3

        Qnew = np.zeros((ns + nt, ns + nt))
        for c in range(C):
            nsc = np.sum(Ys == c)
            nsk = ns - nsc
            ntc = np.sum(Yt_pseudo == c)
            ntk = nt - ntc

            Qck = np.zeros((ns + nt, ns + nt))
            e = np.zeros(ns)
            e[Ys == c] = 1 / nsc
            e[Ys != c] = -1 / nsk
            e[np.isinf(e)] = 0
            Qsck = nsc * np.outer(e, e)

            f = np.zeros(nt)
            f[Yt_pseudo == c] = 1 / ntc
            f[Yt_pseudo != c] = -1 / ntk
            f[np.isinf(f)] = 0
            Qtck = ntc * np.outer(f, f)

            Qck[:ns, :ns] = Qsck
            Qck[ns:, ns:] = Qtck

            g = np.zeros(ns + nt)
            g[:ns][Ys == c] = 1 / nsc
            g[ns + np.where(Yt_pseudo != c)[0]] = -1 / ntk
            g[np.isinf(g)] = 0
            Qstck = nsc * np.outer(g, g)

            h = np.zeros(ns + nt)
            h[:ns][Ys != c] = 1 / nsk
            h[ns + np.where(Yt_pseudo == c)[0]] = -1 / ntc
            h[np.isinf(h)] = 0
            Qtsck = ntc * np.outer(h, h)

            Qnew = Qnew + options["beta"] * Qck + options["gamma"] * Qstck + options["gamma"] * Qtsck

    Qcc = np.zeros((ns + nt, ns + nt))

    tp = csr_matrix((np.ones(ns), (np.arange(ns), Ys)), shape=(ns, C))
    tp2 = tp.multiply(1.0 / np.sum(tp, axis=0))
    tp3 = tp2.dot(tp.T)
    Qcc[:ns, :ns] = np.eye(ns) - tp3.toarray()

    if len(Yt_pseudo) > 0:
        Qcc[ns:, ns:] = Qcc_t

    D = np.diag(np.sum(W, axis=1))
    L = D - W

    Qall = Qcc + M - Qnew

    if simplified:
        Qall += options["eta"] * L

    H = np.eye(n) - 1 / n * np.ones((n, n))

    X = np.vstack((Xs, Xt))
    Omega = X.T @ Qall @ X + options["sigma"] * np.eye(d)
    Omega = (Omega + Omega.T) / 2

    S1 = X.T @ H @ X + 0.0001 * np.eye(d)

    eigenvalues, eigenvectors = eigh(Omega, S1, subset_by_index=[0, options["reduced_dim"] - 1])
    P = eigenvectors[:, :options["reduced_dim"]]

    P = np.real(P)
    for i in range(P.shape[1]):
        if P[0, i] < 0:
            P[:, i] = -P[:, i]

    return P


def CDEM(xs, ys, xt, yt, d, T, simplified, options=None):
    """
    CDEM and simplified CDEM algorithm for unsupervised domain adaptation
    :param xs: features vectors from the source domain
    :param ys: labels from the source domain
    :param xt: features vectors from the target domain
    :param yt: labels from the target domain (only used to print the evolution of the accuracy on target data during adaptation)
    :param d: dimension of the embedding space
    :param T: number of epochs for the training process
    :param simplified: True : simplified CDEM, False : CDEM
    :param options: dictionary of options for CDEM (see reference paper : https://arxiv.org/abs/2106.15057). Proposed default values are : {"beta":1, "lambda":1, "gamma":0.5, "eta":0.0001, "sigma":0.1}
    :return: predicted labels for the target domain samples (xt)
    """
    if options is None:
        options = dict()
    num_iter = T
    options["reduced_dim"] = d
    options["alpha"] = 1

    num_classes = len(np.unique(ys))
    W_all = np.zeros((xs.shape[0] + xt.shape[0],
                      xs.shape[0] + xt.shape[0]))
    W_s = constructW1(ys)
    W = W_all
    W[:W_s.shape[0], :W_s.shape[1]] = W_s

    p = 1
    predLabels = []
    pseudoLabels = []
    for iter in range(1, num_iter + 1):
        try:
            P = constructP(xs, ys, xt, pseudoLabels, W, simplified, options)
            domainS_proj = xs @ P
            domainT_proj = xt @ P
            proj_mean = np.mean(np.concatenate((domainS_proj, domainT_proj), axis=0), axis=0)
            domainS_proj = domainS_proj - proj_mean
            domainT_proj = domainT_proj - proj_mean
            domainS_proj = L2Norm(domainS_proj)
            domainT_proj = L2Norm(domainT_proj)

            classMeans = np.zeros((num_classes, options["reduced_dim"]))
            for i in range(num_classes):
                classMeans[i, :] = np.mean(domainS_proj[ys == i, :], axis=0)

            classMeans = L2Norm(classMeans)
            distClassMeans = EuDist2(domainT_proj, classMeans)

            targetClusterMeans = KMeans(num_classes, n_init=1, init=classMeans).fit(
                domainT_proj).cluster_centers_  # used KMeans instead of vgg_kmeans

            targetClusterMeans = L2Norm(targetClusterMeans)
            distClusterMeans = EuDist2(domainT_proj, targetClusterMeans)
            expMatrix = np.exp(-distClassMeans)
            expMatrix2 = np.exp(-distClusterMeans)
            probMatrix1 = expMatrix / np.sum(expMatrix, axis=1, keepdims=True)
            probMatrix2 = expMatrix2 / np.sum(expMatrix2, axis=1, keepdims=True)

            probMatrix = probMatrix1 * (1 - iter / num_iter) + probMatrix2 * (iter / num_iter)
            prob = np.max(probMatrix.T, axis=0)
            predLabels = np.argmax(probMatrix.T, axis=0)

            I1 = np.argmax(probMatrix1, axis=1)
            I2 = np.argmax(probMatrix2, axis=1)
            samePredict = np.where(I1 == I2)[0]
            prob, predLabels = np.max(probMatrix1, axis=1), np.argmax(probMatrix1, axis=1)
            prob1 = prob[samePredict]
            predLabels1 = predLabels[samePredict]

            p = iter / num_iter
            p = max(p, 0)

            sortedProb = np.sort(prob1)
            index = np.argsort(prob1)
            sortedPredLabels = predLabels1[index]
            trustable = np.zeros(len(prob1))
            for i in range(num_classes):
                ntc = np.sum(predLabels == i)
                ntc_same = np.sum(predLabels1 == i)
                thisClassProb = sortedProb[sortedPredLabels == i]
                if len(thisClassProb) > 0:
                    minProb = thisClassProb[max(ntc_same - (int(np.floor(p * ntc)) + 1), 1) - 1]
                    trustable += (prob1 >= minProb) & (predLabels1 == i)
            trustable = trustable.astype(int)

            print(f"running acc ({iter}):", np.mean(predLabels == yt))

            true_index = samePredict[trustable == 1]
            pseudoLabels = predLabels.copy()
            trustable = np.zeros(len(prob))
            trustable[true_index] = 1
            pseudoLabels[trustable == 0] = -1

            if sum(trustable) >= len(prob):
                break
        except Exception as e:
            print(e)
    return predLabels



if __name__ == "__main__":
    # test of CDEM with synthetic data
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, cross_val_predict
    import pandas as pd
    import seaborn as sn
    from sklearn.preprocessing import StandardScaler, normalize
    from autorank import autorank, plot_stats
    import matplotlib.pyplot as plt

    nSubjects = 10

    x, y = make_classification(random_state=42,
                               n_samples=nSubjects * 200,
                               n_features=20,
                               n_informative=10,
                               n_redundant=5,
                               n_repeated=5,
                               n_classes=3,
                               n_clusters_per_class=1,
                               class_sep=1) # change this value to modify the difficulty of the adaptation
    x = x.reshape((nSubjects, 200, 20))
    y = y.reshape((nSubjects, 200))

    # domain shift
    for i in range(nSubjects):
        for c in range(3):
            x[i][y[i] == c] += (np.random.random(20) * 2 - 1) * 2

    # normalization
    for i in range(len(x)):
        x[i] = normalize(x[i])
        x[i] = StandardScaler().fit_transform(x[i])

    accIntra = []
    accCross = []
    accUDA = []
    for test in range(nSubjects):
        train = np.arange(nSubjects)
        train = np.delete(train, test)

        cl = LogisticRegression("l2", solver="liblinear").fit(np.concatenate(x[train]), np.concatenate(y[train]))
        pred = cl.predict(x[test])

        options = {
            'beta': 1,
            'lambda': 1,
            'gamma': 0.5,
            'eta': 0.0001,
            'sigma': 0.1
        }

        pred2 = CDEM(np.concatenate(x[train]), np.concatenate(y[train]), x[test], y[test], 2, 10, False, options)

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
