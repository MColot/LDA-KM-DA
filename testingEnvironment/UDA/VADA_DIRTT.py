"""
Implementation of the VADA and DIRT-T algorithm in Python, based on their public implementation available on https://github.com/RuiShu/dirt-t
Adaptation of the code to work with custom datasets and using custom neural networks

Dependency : numpy, sklearn, argparse, pathlib, tensorflow, tensorbayes, contextlib
Dependency for testing on synthetic data : pandas, seaborn, matplotlib, autorank

Author : Martin Colot
"""

import tensorflow as tf
from .DIRTT_codebase.datasets_2 import Dataset
from .DIRTT_codebase.models.dirtt import dirtt
from .DIRTT_codebase.train import train
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pathlib
import multiprocessing

def run_VADA_DIRTT(xs, ys, xt, yt, nclasses, saveDirectory, dirt = 0, lr = 0.001, tw=0.01, epochs=10, batchPerEpoch=1000, multiProcessing=False):
    """
    Runs VADA or DIRT-T in a separate process if necessary to ensure right version of tensorflow when other library are used
    """
    if multiProcessing:
        queue = multiprocessing.Queue()
        p1 = multiprocessing.Process(target=VADA_DIRTT, args=(xs, ys, xt, yt, nclasses, saveDirectory, dirt, lr, tw, epochs, batchPerEpoch, queue))
        print("starting subprocess (this operation might take time)")
        p1.start()
        p1.join()
        return queue.get()
    else:
        return VADA_DIRTT(xs, ys, xt, yt, nclasses, saveDirectory, dirt, lr, tw, epochs, batchPerEpoch)

def VADA_DIRTT(xs, ys, xt, yt, nclasses, saveDirectory, dirt = 0, lr = 0.001, tw=0.01, epochs=10, batchPerEpoch=1000, queue=None):
    """

    :param xs: features vectors from the source domain
    :param ys: labels from the source domain
    :param xt: features vectors from the target domain
    :param yt: labels from the target domain (used to print accuracy only)
    :param nclasses: number of classes
    :param saveDirectory: directory to save the trained classifier (useful when DIRT-T is used after VADA)
    :param dirt: number of batches before updating the teacher (DIRT-T is implemented as a recursive VADA)
    :param lr: learning rate
    :param tw: target weight = how much the classifier should be stable on target samples with respect to small perturbations
    :param epochs: number of epochs
    :param batchPerEpoch : number of batches per epoch
    :param queue : results for multiprocessing
    :return: prediction of labels of the target samples (xt)
    """
    tf.compat.v1.disable_eager_execution()

    ohe = OneHotEncoder(sparse_output=False).fit(ys.reshape(-1, 1))
    ys = ohe.transform(ys.reshape(-1, 1))
    yt = ohe.transform(yt.reshape(-1, 1))

    args = dict()
    args["inorm"] = 1
    args["radius"] = 3.5
    args["dw"] = 1e-2 # importance of adversarial loss between source and target (only in VADA)
    args["bw"] = 1e-2  # importance of classifier error on sources in DIRT-T (for VADA, bw is set to 1)
    args["sw"] = 0 # how much the classifier should be stable on sources sable with respect to small perturbations (only during VADA)
    args["tw"] = tw  # how much the classifier should be stable on target samples with respect to small perturbations
    args["run"] = 999
    args["logdir"] = "log"

    args["src"] = "custom"
    args["trg"] = "custom"

    #custom parameters
    args["datadir"] = "data"
    args["dirt"] = dirt #VADA : 0, DIRT-T : k (steps before update teacher)
    args["nn"] = "simple"
    args["trim"] = 1
    args["lr"] = lr
    args["Y"] = nclasses

    # Argument overrides and additions
    args["H"] = 32
    args["bw"] = args["bw"] if args["dirt"] > 0 else 0.  # mask bw when running VADA


    # Make model name
    setup = [
        ('model={:s}',  'dirtt'),
        ('src={:s}',    args["src"]),
        ('trg={:s}',    args["trg"]),
        ('nn={:s}',     args["nn"]),
        ('trim={:d}',   args["trim"]),
        ('dw={:.0e}',   args["dw"]),
        ('bw={:.0e}',   args["bw"]),
        ('sw={:.0e}',   args["sw"]),
        ('tw={:.0e}',   args["tw"]),
        ('dirt={:05d}', args["dirt"]),
        ('run={:04d}',  args["run"])
    ]
    model_name = '_'.join([t.format(v) for (t, v) in setup])
    print ("Model name:", model_name)

    M = dirtt(xShape=xs.shape[1], args=args)
    M.sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    if args["dirt"] > 0:
        path = tf.train.latest_checkpoint(saveDirectory)
        saver.restore(M.sess, path)

    src = Dataset(xs, ys)
    trg = Dataset(xt, yt)

    train(M, saveDirectory, src, trg, has_disc=args["dirt"] == 0, iterep=batchPerEpoch, n_epoch=epochs, bs=64, saver=saver, args=args)
    pred = M.teacher(trg.train.images)

    if queue is not None:
        queue.put(np.argmax(pred, axis=1))
    return np.argmax(pred, axis=1)





if __name__ == "__main__":
    # test of VADA and DIRT-T with synthetic data
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
                               class_sep=0.6) # change this value to modify the difficulty of the adaptation
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
    accVADA = []
    accDIRTT = []
    for test in range(nSubjects):
        train = np.arange(nSubjects)
        train = np.delete(train, test)

        cl = LogisticRegression("l2", solver="liblinear").fit(np.concatenate(x[train]), np.concatenate(y[train]))
        pred = cl.predict(x[test])

        saveDirectory = str(pathlib.Path(__file__).parent.resolve()) + "/DIRTT_codebase/save"

        lr = 0.01
        tf.compat.v1.reset_default_graph()
        predVada = VADA_DIRTT(np.concatenate(x[train]), np.concatenate(y[train]), x[test], y[test], 3, saveDirectory, 0, lr, 0.01, 10, 300)
        tf.compat.v1.reset_default_graph()
        predDirtt = VADA_DIRTT(np.concatenate(x[train]), np.concatenate(y[train]), x[test], y[test], 3, saveDirectory, 600, lr, 0.01, 10, 300)

        accCross.append(np.mean(y[test] == pred))
        accVADA.append(np.mean(y[test] == predVada))
        accDIRTT.append(np.mean(y[test] == predDirtt))

        predIntra = cross_val_predict(LogisticRegression("l2", solver="liblinear"), x[test], y[test],
                                      cv=KFold(n_splits=5, shuffle=False))
        accIntra.append(np.mean(y[test] == predIntra))

        print(accCross[-1], " -> ", accVADA[-1], " -> ", accDIRTT[-1], " / ", accIntra[-1])

    scoreDF = pd.DataFrame(
        {"Cross-subjects\nwith normalization": accCross, "VADA": accVADA, "DIRT-T": accDIRTT, "Intra-subject": accIntra})
    sn.boxplot(scoreDF)
    plt.show()

    result = autorank(scoreDF, alpha=0.05, verbose=False)
    plot_stats(result)
    plt.show()
