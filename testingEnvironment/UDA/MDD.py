"""
Custom encoder and decoder neural networks, and training parameters for the MDD method
Using the MDD implementation available in the ADAPT package (https://adapt-python.github.io/adapt/index.html)

Dependency : sklearn, numpy, tensorflow, tf_keras, adapt
Dependency for testing on synthetic data : pandas, seaborn, matplotlib, autorank

Author : Martin Colot
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = 'True'
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from adapt.feature_based import MDD


def custumEncoder(input_size, hidden_size):
    """
    Custom encoder neural network for MDD
    :param input_size: shape of the input vectors
    :param hidden_size: number of neurons in the hidden layer
    :return: tf neural network
    """
    inputs = Input(shape=(input_size,))
    x = layers.Dense(hidden_size, activation='sigmoid')(inputs)
    x = layers.BatchNormalization()(x)
    model = Model(inputs=inputs, outputs=x)
    return model

def customDiscriminator(hidden_size, num_classes):
    """
    Custom decoder neural network for MDD
    :param hidden_size: number of neurons in the hidden layer
    :param num_classes : number of classes
    :return: tf neural network
    """
    model = tf.keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(hidden_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def fit_MDD(xs, ys, xt, hidden=10, epochs=20):
    """
    Training of MDD model using custom parameters and neural networks
    :param xs: features vectors from source domain
    :param ys: labels from source domain
    :param xt: fearures vectors from target domain
    :param hidden: size of the embedding
    :param epochs: number of epochs to train the MDD model
    :return: trained classifier
    """

    nclasses = len(np.unique(ys))
    ohe = OneHotEncoder().fit(ys.reshape(-1, 1))
    ys = ohe.transform(ys.reshape(-1, 1)).toarray()
    G = custumEncoder(xs.shape[1], hidden)
    F = customDiscriminator(hidden, nclasses)
    classifier = MDD(encoder = G, task=F, Xt = xt, metrics=["acc"], loss="categorical_crossentropy",
                     optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, beta_1=0.5))
    classifier.fit(xs, ys, epochs=epochs, batch_size=128)
    return classifier



if __name__ == "__main__":
    # test of MDD with synthetic data
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, cross_val_predict
    import pandas as pd
    import seaborn as sn
    from sklearn.preprocessing import StandardScaler, normalize
    from autorank import autorank, plot_stats
    import matplotlib.pyplot as plt

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

    accIntra = []
    accCross = []
    accUDA = []
    for test in range(nSubjects):
        train = np.arange(nSubjects)
        train = np.delete(train, test)

        cl = LogisticRegression("l2", solver="liblinear").fit(np.concatenate(x[train]), np.concatenate(y[train]))
        pred = cl.predict(x[test])

        mdd = fit_MDD(np.concatenate(x[train]), np.concatenate(y[train]), x[test], 20, 50)
        pred2 = np.argmax(mdd.predict(x[test]), axis=1)

        accCross.append(np.mean(y[test] == pred))
        accUDA.append(np.mean(y[test] == pred2))

        predIntra = cross_val_predict(LogisticRegression("l2", solver="liblinear"), x[test], y[test],
                                      cv=KFold(n_splits=5, shuffle=False))
        accIntra.append(np.mean(y[test] == predIntra))

        print(accCross[-1], " -> ", accUDA[-1], " / ", accIntra[-1])

    scoreDF = pd.DataFrame(
        {"Cross-subjects\nwith normalization": accCross, "MDD": accUDA, "Intra-subject": accIntra})
    sn.boxplot(scoreDF)
    plt.show()

    result = autorank(scoreDF, alpha=0.05, verbose=False)
    plot_stats(result)
    plt.show()
