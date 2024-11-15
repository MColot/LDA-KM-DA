import numpy as np

class Data(object):
    def __init__(self, images, labels=None, labeler=None, cast=False):
        """Data object constructs mini-batches to be fed during training

        images - (NHWC) data
        labels - (NK) one-hot data
        labeler - (tb.function) returns simplex value given an image
        cast - (bool) converts uint8 to [-1, 1] float
        """
        self.images = images
        self.labels = labels
        self.labeler = labeler
        self.cast = cast

    def next_batch(self, bs):
        """Constructs a mini-batch of size bs without replacement
        """
        idx = np.random.choice(len(self.images), bs, replace=False)
        x = self.images[idx]
        y = self.labeler(x) if self.labels is None else self.labels[idx]
        return x, y


class Dataset(object):
    def __init__(self, x, y):
        self.train = Data(x, y)
        self.test = Data(x, y)


class PseudoData(object):
    def __init__(self, domain_id, domain, teacher):
        """Variable domain with pseudolabeler

        domain_id - (str) {Mnist, Mnistm, Svhn, etc}
        domain - (obj) {Mnist, Mnistm, Svhn, etc}
        teacher - (fn) Teacher model used for pseudolabeling
        """
        print("Constructing pseudodata")
        cast = 'mnist' not in domain_id
        print("{} uses casting: {}".format(domain_id, cast))
        labeler = teacher

        self.train = Data(domain.train.images, labeler=labeler, cast=cast)
        self.test = Data(domain.test.images, labeler=labeler, cast=cast)